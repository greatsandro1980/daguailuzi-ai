/**
 * 大怪路子多人联机服务器
 * 使用 WebSocket（ws 库）管理房间、席位、游戏状态
 */

import { WebSocketServer, WebSocket } from 'ws';
import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, extname, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
import {
  createDeck, shuffleDeck, dealCards, sortCards,
  recognizeCardType, compareCards, aiSelectCards,
  countJokersInHand, getJokerCards, getBestCard, getBestCards,
  getCardDisplay, CardType
} from './gameLogic.js';

const PORT = process.env.PORT || 3002;

// ─── 游戏常量 ─────────────────────────────────────────
const SEAT_COUNT = 6;
// 座位 -> 队伍映射：0,2,4=红队(team=1)  1,3,5=蓝队(team=2)
const SEAT_TEAM = [1, 2, 1, 2, 1, 2];
const SEAT_LABELS = ['红队1号', '蓝队1号', '红队2号', '蓝队2号', '红队3号', '蓝队3号'];

// AI 角色列表
const AI_CHARACTERS = [
  { id: 'zhiling',  name: '志玲',  emoji: '💃' },
  { id: 'dami',     name: '大幂',  emoji: '🌟' },
  { id: 'yuanyuan', name: '圆圆',  emoji: '🌸' },
  { id: 'dehua',    name: '德华',  emoji: '🎤' },
  { id: 'chengwu',  name: '城武',  emoji: '🎬' },
  { id: 'yanzhu',   name: '彦祖',  emoji: '😎' },
];

const GamePhase = {
  LOBBY: 'lobby',
  DEALING: 'dealing',
  TEAMING: 'teaming',
  PLAYING: 'playing',
  ROUND_END: 'round_end',
  TRIBUTE: 'tribute',
  TRIBUTE_RETURN: 'tribute_return',
  GAME_OVER: 'game_over'
};

// ─── 房间状态 ─────────────────────────────────────────
let room = createRoom();

function createRoom() {
  return {
    // 大厅席位：length=6，null=空，否则={clientId, nickname}
    seats: Array(SEAT_COUNT).fill(null),
    // 就绪状态
    readySeats: new Set(),
    // 游戏状态
    game: null,
    // 消息日志
    messages: []
  };
}

function createGame() {
  return {
    phase: GamePhase.DEALING,
    players: [],       // Player[6]，index 对应 seat
    currentPlayer: 0,  // seatIndex
    selectedCards: [], // 本轮已出牌（服务器记录，用于广播）
    playedCards: [],   // [{seatIndex, cards}] 桌面牌
    playerShowCards: {},
    currentRoundCardCount: null,
    roundStarter: null,
    roundWinner: null,
    firstPlay: true,
    jokers: [],
    trumpRank: null,
    finishOrder: [],
    scores: { red: 0, blue: 0 },
    dealerTeam: 1,
    lastWinnerId: null,
    isFirstRound: true,
    redLevel: 2,
    blueLevel: 2,
    tributeInfo: null,
    pendingTributeInfo: null,
    lastTributeInfo: [],
    countdown: 15,
    tributeCountdown: 15,
    returnCountdown: 15,
    tributeForceNoResist: false,
    selectedReturnCandidates: [],
    countdownTimer: null,   // 服务端倒计时 interval（不广播）
  };
}

// ─── WebSocket 服务器 ─────────────────────────────────
// 静态文件目录（前端打包产物）
const STATIC_DIR = join(__dirname, '../daguailuzi/dist');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.json': 'application/json',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
};

const httpServer = createServer((req, res) => {
  const url = new URL(req.url || '/', `http://localhost`);
  // 解码 URL 中的中文文件名
  const decodedPathname = decodeURIComponent(url.pathname);
  let filePath = join(STATIC_DIR, decodedPathname);

  // 如果不是静态文件目录，回退到 index.html（SPA 路由）
  if (!existsSync(filePath) || existsSync(filePath) && filePath.endsWith('/')) {
    filePath = join(STATIC_DIR, 'index.html');
  }

  if (existsSync(filePath)) {
    const ext = extname(filePath).toLowerCase();
    const mime = MIME_TYPES[ext] || 'application/octet-stream';
    try {
      const content = readFileSync(filePath);
      res.writeHead(200, { 'Content-Type': mime });
      res.end(content);
      return;
    } catch {}
  }

  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('大怪路子游戏服务器运行中');
});

const wss = new WebSocketServer({ server: httpServer });

// clientId -> WebSocket
const clients = new Map();
// clientId -> {nickname, seatIndex (-1=观战), playerId}
const clientInfo = new Map();

// 断线玩家缓存：playerId -> {nickname, seatIndex, hand, reconnectDeadline}
// reconnectDeadline 是断线时间 + 5分钟的毫秒时间戳
const disconnectedPlayers = new Map();

// playerId -> clientId 的映射（用于重连时查找）
const playerIdToClientId = new Map();

let clientIdCounter = 1;

// 断线重连超时时间（5分钟）
const RECONNECT_TIMEOUT_MS = 5 * 60 * 1000;

// 定期清理过期的断线玩家缓存
setInterval(() => {
  const now = Date.now();
  for (const [playerId, data] of disconnectedPlayers.entries()) {
    if (now > data.reconnectDeadline) {
      console.log(`[清理] 玩家 ${data.nickname}(${playerId}) 断线超时，清除数据`);
      disconnectedPlayers.delete(playerId);
      // 如果游戏还在进行，需要处理该玩家彻底离开
      if (data.wasInGame && room.game) {
        handlePlayerTimeout(playerId, data);
      }
    }
  }
}, 30000); // 每30秒检查一次

wss.on('connection', (ws) => {
  const clientId = String(clientIdCounter++);
  clients.set(clientId, ws);
  console.log(`[连接] 客户端 ${clientId} 加入`);

  // 发送欢迎消息和当前房间状态
  send(ws, { type: 'welcome', clientId });
  sendRoomState(ws);

  ws.on('message', (raw) => {
    try {
      const msg = JSON.parse(raw.toString());
      handleMessage(clientId, msg);
    } catch (e) {
      console.error('消息解析失败', e);
    }
  });

  ws.on('close', () => {
    console.log(`[断开] 客户端 ${clientId} 离开`);
    handleDisconnect(clientId);
  });
});

httpServer.listen(PORT, () => {
  console.log(`大怪路子服务器启动：端口 ${PORT}`);
});

// ─── 消息发送工具 ─────────────────────────────────────
function send(ws, data) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(data));
  }
}

function sendTo(clientId, data) {
  const ws = clients.get(clientId);
  if (ws) send(ws, data);
}

function broadcast(data, excludeClientId = null) {
  for (const [cid, ws] of clients.entries()) {
    if (cid !== excludeClientId) send(ws, data);
  }
}

// ─── 状态广播 ─────────────────────────────────────────
/**
 * 构建公开的房间快照（隐藏手牌）
 */
function buildRoomSnapshot() {
  const seatsInfo = room.seats.map((s, i) => ({
    seatIndex: i,
    label: SEAT_LABELS[i],
    team: SEAT_TEAM[i],
    occupied: !!s,
    clientId: s?.clientId || null,
    nickname: s?.nickname || null,
    ready: room.readySeats.has(i),
    disconnected: s?.disconnected || false,
    playerId: s?.playerId || undefined,
    isAI: s?.isAI || false,
    aiId: s?.aiId || null,
  }));

  let gamePublic = null;
  if (room.game) {
    const g = room.game;
    gamePublic = {
      phase: g.phase,
      players: g.players.map(p => ({
        seatIndex: p.seatIndex,
        name: p.name,
        team: p.team,
        isOut: p.isOut,
        cardCount: p.cardCount,
        hasAsked: p.hasAsked,
        disconnected: room.seats[p.seatIndex]?.disconnected || false
        // hand 不暴露
      })),
      currentPlayer: g.currentPlayer,
      playedCards: g.playedCards,
      playerShowCards: g.playerShowCards,
      currentRoundCardCount: g.currentRoundCardCount,
      roundStarter: g.roundStarter,
      roundWinner: g.roundWinner,
      firstPlay: g.firstPlay,
      jokers: g.jokers,
      trumpRank: g.trumpRank,
      finishOrder: g.finishOrder,
      scores: g.scores,
      dealerTeam: g.dealerTeam,
      lastWinnerId: g.lastWinnerId,
      isFirstRound: g.isFirstRound,
      redLevel: g.redLevel,
      blueLevel: g.blueLevel,
      tributeInfo: buildPublicTributeInfo(g.tributeInfo),
      lastTributeInfo: g.lastTributeInfo,
      countdown: g.countdown,
      tributeCountdown: g.tributeCountdown,
      returnCountdown: g.returnCountdown,
      tributeForceNoResist: g.tributeForceNoResist,
      selectedReturnCandidates: g.selectedReturnCandidates
    };
  }

  return {
    seats: seatsInfo,
    game: gamePublic,
    messages: room.messages.slice(-20),
    aiCharacters: AI_CHARACTERS,
    usedAIIds: getUsedAIIds(),
  };
}

function buildPublicTributeInfo(ti) {
  if (!ti) return null;
  return {
    tributors: ti.tributors,
    receivers: ti.receivers,
    currentTributorIndex: ti.currentTributorIndex,
    currentReceiverIndex: ti.currentReceiverIndex,
    tributes: ti.tributes,
    returns: ti.returns,
    canResist: ti.canResist,
    resistedPlayers: ti.resistedPlayers,
    jokerCards: ti.jokerCards,
    returnSubPhase: ti.returnSubPhase,
    returnCandidates: ti.returnCandidates
  };
}

function sendRoomState(ws) {
  send(ws, { type: 'room_state', data: buildRoomSnapshot() });
}

function broadcastRoomState() {
  const snapshot = buildRoomSnapshot();
  const msg = JSON.stringify({ type: 'room_state', data: snapshot });
  for (const ws of clients.values()) {
    if (ws.readyState === WebSocket.OPEN) ws.send(msg);
  }
}

/**
 * 给指定客户端发送其私有手牌
 */
function sendHandToPlayer(clientId) {
  const info = clientInfo.get(clientId);
  if (!info || info.seatIndex < 0 || !room.game) return;
  const player = room.game.players[info.seatIndex];
  if (!player) return;
  sendTo(clientId, { type: 'your_hand', hand: player.hand, sortedCards: player.sortedCards });
}

function broadcastAllHands() {
  for (const [cid, info] of clientInfo.entries()) {
    if (info && info.seatIndex >= 0) sendHandToPlayer(cid);
  }
}

function addMessage(text, type = 'info') {
  room.messages.push({ id: Date.now(), text, type, timestamp: Date.now() });
  if (room.messages.length > 100) room.messages.shift();
}

// ─── 消息处理 ─────────────────────────────────────────
function handleMessage(clientId, msg) {
  console.log(`[消息] ${clientId} -> ${msg.type}`);

  switch (msg.type) {
    case 'set_player_id':
      handleSetPlayerId(clientId, msg);
      break;
    case 'set_nickname':
      handleSetNickname(clientId, msg);
      break;
    case 'take_seat':
      handleTakeSeat(clientId, msg);
      break;
    case 'leave_seat':
      handleLeaveSeat(clientId);
      break;
    case 'add_ai':
      handleAddAI(clientId, msg);
      break;
    case 'remove_ai':
      handleRemoveAI(clientId, msg);
      break;
    case 'toggle_ready':
      handleToggleReady(clientId);
      break;
    case 'select_trump':
      handleSelectTrump(clientId, msg);
      break;
    case 'play_cards':
      handlePlayCards(clientId, msg);
      break;
    case 'pass_play':
      handlePassPlay(clientId);
      break;
    case 'start_next_round':
      handleStartNextRound(clientId);
      break;
    case 'tribute_resist':
      handleTributeResist(clientId, msg);
      break;
    case 'tribute_confirm':
      handleTributeConfirm(clientId, msg);
      break;
    case 'return_candidates_confirm':
      handleReturnCandidatesConfirm(clientId, msg);
      break;
    case 'return_confirm':
      handleReturnConfirm(clientId, msg);
      break;
    default:
      console.warn('未知消息类型:', msg.type);
  }
}

function handleDisconnect(clientId) {
  const info = clientInfo.get(clientId);
  if (info) {
    const { nickname, seatIndex, playerId } = info;
    
    // 如果玩家在游戏中，缓存断线数据
    if (room.game && seatIndex >= 0 && room.seats[seatIndex]?.clientId === clientId) {
      const player = room.game.players[seatIndex];
      console.log(`[断线] ${nickname}(${playerId}) 在游戏中断线，开始5分钟重连倒计时`);
      
      // 缓存断线玩家数据
      disconnectedPlayers.set(playerId, {
        nickname,
        seatIndex,
        hand: player ? [...player.hand] : [],
        wasInGame: true,
        reconnectDeadline: Date.now() + RECONNECT_TIMEOUT_MS,
        // 保存游戏中的其他状态
        finished: player?.finished || false
      });
      
      // 标记座位为断线状态，但不清空
      room.seats[seatIndex] = {
        ...room.seats[seatIndex],
        disconnected: true,
        playerId
      };
      
      // 移除准备状态
      room.readySeats.delete(seatIndex);
      
      addMessage(`${nickname} 断线了（5分钟内可重连）`, 'warning');
      
      // 如果当前是断线玩家的回合，可能需要暂停或跳过
      if (room.game.currentPlayer === seatIndex) {
        // TODO: 可以选择暂停倒计时或自动跳过
        // 这里选择暂停倒计时
        stopCountdown(room.game);
        addMessage(`等待 ${nickname} 重连...`, 'system');
      }
    } else if (seatIndex >= 0 && room.seats[seatIndex]?.clientId === clientId) {
      // 大厅中断线，直接清空座位
      room.seats[seatIndex] = null;
      room.readySeats.delete(seatIndex);
      addMessage(`${nickname} 离开了游戏`, 'warning');
    }
    
    clientInfo.delete(clientId);
    playerIdToClientId.delete(playerId);
  }
  clients.delete(clientId);
  broadcastRoomState();
}

// 玩家超时未重连的处理
function handlePlayerTimeout(playerId, data) {
  const { nickname, seatIndex } = data;
  console.log(`[超时] ${nickname}(${playerId}) 超时未重连`);
  
  // 判断断线玩家所在队伍
  const timeoutTeam = SEAT_TEAM[seatIndex]; // 1=红队, 2=蓝队
  const teamName = timeoutTeam === 1 ? '红方' : '蓝方';
  
  addMessage(`${nickname} 超时未重连，${teamName}被判负！`, 'error');
  
  // 对方获胜，计算得分（对方加3分，因为对方3人全部获胜）
  const winnerTeam = timeoutTeam === 1 ? 2 : 1;
  const winnerName = winnerTeam === 1 ? '红方' : '蓝方';
  
  // 更新得分
  if (room.game) {
    if (winnerTeam === 1) {
      room.game.scores.red += 3;
    } else {
      room.game.scores.blue += 3;
    }
    addMessage(`${winnerName} 获得 3 分！`, 'success');
  }
  
  // 广播最终状态
  broadcastRoomState();
  
  // 3秒后重置游戏，踢出所有玩家
  setTimeout(() => {
    resetGameAndKickAll(`${teamName} 有玩家超时未重连，游戏结束`);
  }, 3000);
}

// 重置游戏并踢出所有玩家
function resetGameAndKickAll(reason) {
  console.log(`[重置] ${reason}`);
  
  // 通知所有玩家游戏结束
  broadcast({ 
    type: 'game_reset', 
    reason,
    message: '游戏已重置，请重新加入'
  });
  
  // 清空所有客户端信息
  for (const [clientId, ws] of clients.entries()) {
    try {
      ws.close();
    } catch (e) {}
  }
  clients.clear();
  clientInfo.clear();
  playerIdToClientId.clear();
  disconnectedPlayers.clear();
  
  // 重置房间
  room = createRoom();
  addMessage(reason, 'system');
  
  console.log('[重置] 游戏已重置，所有玩家已踢出');
}

// ─── 大厅逻辑 ─────────────────────────────────────────

// 处理玩家设置 playerId（用于断线重连）
function handleSetPlayerId(clientId, msg) {
  const playerId = msg.playerId;
  if (!playerId) return;
  
  const existing = clientInfo.get(clientId) || { seatIndex: -1, nickname: '' };
  
  // 检查是否有断线的玩家数据
  const disconnectedData = disconnectedPlayers.get(playerId);
  
  if (disconnectedData && Date.now() <= disconnectedData.reconnectDeadline) {
    // 找到断线数据，进行重连
    console.log(`[重连] 客户端 ${clientId} 重连为 ${disconnectedData.nickname}(${playerId})`);
    
    // 恢复玩家信息
    const seatIndex = disconnectedData.seatIndex;
    clientInfo.set(clientId, {
      ...existing,
      nickname: disconnectedData.nickname,
      seatIndex: seatIndex,
      playerId: playerId
    });
    
    playerIdToClientId.set(playerId, clientId);
    
    // 更新座位信息
    if (room.seats[seatIndex]?.playerId === playerId) {
      room.seats[seatIndex] = {
        clientId: clientId,
        nickname: disconnectedData.nickname,
        playerId: playerId,
        disconnected: false  // 标记为已重连
      };
      
      // 如果游戏还在进行，恢复手牌
      if (room.game && room.game.players[seatIndex]) {
        room.game.players[seatIndex].hand = disconnectedData.hand;
        room.game.players[seatIndex].finished = disconnectedData.finished;
        room.game.players[seatIndex].timeout = false;
      }
    }
    
    // 清除断线缓存
    disconnectedPlayers.delete(playerId);
    
    // 发送重连成功消息
    sendTo(clientId, {
      type: 'reconnect_success',
      nickname: disconnectedData.nickname,
      seatIndex: seatIndex,
      hand: disconnectedData.hand,
      game: room.game ? buildGameSnapshotForPlayer(seatIndex) : null
    });
    
    addMessage(`${disconnectedData.nickname} 已重连`, 'success');
    
    // 如果当前是重连玩家的回合，恢复倒计时
    if (room.game && room.game.currentPlayer === seatIndex) {
      startCountdown(room.game);
    }
    
    broadcastRoomState();
  } else {
    // 没有断线数据，只是记录 playerId
    clientInfo.set(clientId, { ...existing, playerId });
    playerIdToClientId.set(playerId, clientId);
    
    // 告知客户端 playerId 已记录
    sendTo(clientId, { type: 'player_id_ok', playerId });
  }
}

// 构建给特定玩家的游戏快照（包含手牌）
function buildGameSnapshotForPlayer(seatIndex) {
  if (!room.game) return null;
  const g = room.game;
  return {
    phase: g.phase,
    players: g.players.map((p, i) => ({
      nickname: p.nickname,
      handCount: p.hand.length,
      finished: p.finished,
      timeout: p.timeout,
      // 只给自己的手牌
      hand: i === seatIndex ? p.hand : undefined
    })),
    currentPlayer: g.currentPlayer,
    playedCards: g.playedCards,
    playerShowCards: g.playerShowCards,
    currentRoundCardCount: g.currentRoundCardCount,
    firstPlay: g.firstPlay,
    jokers: g.jokers,
    trumpRank: g.trumpRank,
    finishOrder: g.finishOrder,
    scores: g.scores,
    dealerTeam: g.dealerTeam,
    redLevel: g.redLevel,
    blueLevel: g.blueLevel,
    tributeInfo: g.tributeInfo,
    countdown: g.countdown
  };
}

function handleSetNickname(clientId, msg) {
  const nickname = String(msg.nickname || '').trim().slice(0, 12);
  if (!nickname) return;
  const existing = clientInfo.get(clientId);
  clientInfo.set(clientId, { ...(existing || { seatIndex: -1 }), nickname });
  sendTo(clientId, { type: 'nickname_ok', nickname });
}

function handleTakeSeat(clientId, msg) {
  const info = clientInfo.get(clientId);
  if (!info || !info.nickname) {
    sendTo(clientId, { type: 'error', message: '请先设置昵称' });
    return;
  }
  if (room.game) {
    sendTo(clientId, { type: 'error', message: '游戏已在进行中，请观战' });
    return;
  }
  const seatIndex = Number(msg.seatIndex);
  if (seatIndex < 0 || seatIndex >= SEAT_COUNT) return;
  if (room.seats[seatIndex]) {
    sendTo(clientId, { type: 'error', message: '该座位已被占用' });
    return;
  }
  // 离开旧座位
  if (info.seatIndex >= 0) {
    room.seats[info.seatIndex] = null;
    room.readySeats.delete(info.seatIndex);
  }
  room.seats[seatIndex] = { clientId, nickname: info.nickname };
  clientInfo.set(clientId, { ...info, seatIndex });
  addMessage(`${info.nickname} 入座 ${SEAT_LABELS[seatIndex]}`, 'info');
  broadcastRoomState();
}

function handleLeaveSeat(clientId) {
  const info = clientInfo.get(clientId);
  if (!info || info.seatIndex < 0) return;
  if (room.game) {
    sendTo(clientId, { type: 'error', message: '游戏进行中不能离座' });
    return;
  }
  room.seats[info.seatIndex] = null;
  room.readySeats.delete(info.seatIndex);
  clientInfo.set(clientId, { ...info, seatIndex: -1 });
  addMessage(`${info.nickname} 离开了座位`, 'info');
  broadcastRoomState();
}

// 获取当前所有座位已使用的 AI 角色 id 列表
function getUsedAIIds() {
  return room.seats
    .filter(s => s && s.isAI)
    .map(s => s.aiId);
}

function handleAddAI(clientId, msg) {
  if (room.game) {
    sendTo(clientId, { type: 'error', message: '游戏进行中无法添加AI' });
    return;
  }
  const seatIndex = Number(msg.seatIndex);
  if (seatIndex < 0 || seatIndex >= SEAT_COUNT) return;
  if (room.seats[seatIndex]) {
    sendTo(clientId, { type: 'error', message: '该座位已被占用' });
    return;
  }
  const aiId = msg.aiId;
  const character = AI_CHARACTERS.find(c => c.id === aiId);
  if (!character) {
    sendTo(clientId, { type: 'error', message: '无效的AI角色' });
    return;
  }
  if (getUsedAIIds().includes(aiId)) {
    sendTo(clientId, { type: 'error', message: `${character.name} 已在其他座位` });
    return;
  }
  // 占座并标记为AI，自动就绪
  room.seats[seatIndex] = {
    clientId: `ai_${aiId}`,
    nickname: `${character.emoji}${character.name}`,
    isAI: true,
    aiId,
    ready: true,
  };
  room.readySeats.add(seatIndex);
  addMessage(`${character.emoji}${character.name} 加入了 ${SEAT_LABELS[seatIndex]}（AI陪打）`, 'info');
  broadcastRoomState();

  // 检查是否 6 人全就绪
  checkAllReady();
}

function handleRemoveAI(clientId, msg) {
  if (room.game) {
    sendTo(clientId, { type: 'error', message: '游戏进行中无法移除AI' });
    return;
  }
  const seatIndex = Number(msg.seatIndex);
  if (seatIndex < 0 || seatIndex >= SEAT_COUNT) return;
  const seat = room.seats[seatIndex];
  if (!seat || !seat.isAI) return;
  const name = seat.nickname;
  room.seats[seatIndex] = null;
  room.readySeats.delete(seatIndex);
  addMessage(`${name} 离开了 ${SEAT_LABELS[seatIndex]}`, 'info');
  broadcastRoomState();
}

function checkAllReady() {
  const allOccupied = room.seats.every(s => s !== null);
  const allReady = allOccupied && room.readySeats.size === SEAT_COUNT;
  if (allReady) {
    setTimeout(() => startGame(), 1000);
  }
}

function handleToggleReady(clientId) {
  const info = clientInfo.get(clientId);
  if (!info || info.seatIndex < 0) return;
  if (room.game) return;
  const si = info.seatIndex;
  if (room.readySeats.has(si)) {
    room.readySeats.delete(si);
  } else {
    room.readySeats.add(si);
  }
  broadcastRoomState();

  // 检查是否 6 人全就绪
  checkAllReady();
}

// ─── 游戏开始 ─────────────────────────────────────────
function startGame() {
  // 再次校验
  if (!room.seats.every(s => s !== null)) return;

  const g = createGame();
  room.game = g;

  // 创建玩家
  g.players = room.seats.map((seat, i) => ({
    seatIndex: i,
    clientId: seat.clientId,
    name: seat.nickname,
    team: SEAT_TEAM[i],
    hand: [],
    sortedCards: [],
    isOut: false,
    cardCount: 0,
    hasAsked: false
  }));

  // 第一盘：随机抽取一位作为首轮发牌者
  const firstDealer = Math.floor(Math.random() * SEAT_COUNT);
  g.lastWinnerId = firstDealer; // 发牌者也是第一个出牌的人
  g.dealerTeam = SEAT_TEAM[firstDealer]; // 发牌者所在队伍成为庄家
  addMessage(`随机抽取：${g.players[firstDealer].name} 成为首轮发牌者，${g.dealerTeam === 1 ? '红方' : '蓝方'}为庄家`, 'info');

  // 发牌
  const deck = shuffleDeck(createDeck());
  const hands = dealCards(deck, SEAT_COUNT);
  const allJokers = deck.filter(c => c.isJoker);
  g.jokers = allJokers;

  // 直接设定将牌：根据庄家队的级别（第一轮默认打2）
  const trumpRank = g.dealerTeam === 1 ? g.redLevel : g.blueLevel;
  g.trumpRank = trumpRank;

  for (let i = 0; i < SEAT_COUNT; i++) {
    g.players[i].hand = hands[i];
    g.players[i].sortedCards = sortCards(hands[i], allJokers, trumpRank);
    g.players[i].cardCount = hands[i].length;
  }

  // 直接进入游戏阶段
  g.phase = GamePhase.PLAYING;
  g.firstPlay = true;
  g.currentPlayer = firstDealer; // 第一盘从发牌者开始

  const trumpName = trumpRank === 14 ? 'A' : trumpRank === 13 ? 'K' : trumpRank === 12 ? 'Q' : trumpRank === 11 ? 'J' : String(trumpRank);
  addMessage(`游戏开始！${g.dealerTeam === 1 ? '红方' : '蓝方'}打${trumpName}`, 'success');

  broadcastRoomState();
  broadcastAllHands();

  // 如果当前玩家是 AI，触发 AI 出牌
  scheduleAIPlayIfNeeded(g);
  startPlayCountdown(g);
}

// ─── 将牌选择 ─────────────────────────────────────────
function handleSelectTrump(clientId, msg) {
  const info = clientInfo.get(clientId);
  if (!info || !room.game) return;
  const g = room.game;
  if (g.phase !== GamePhase.TEAMING) return;
  if (info.seatIndex !== g.currentPlayer) {
    sendTo(clientId, { type: 'error', message: '不是你选将牌' });
    return;
  }
  const rank = Number(msg.rank);
  if (rank < 2 || rank > 14) return;
  applyTrumpSelection(g, rank);
}

function handleSelectTrumpAuto(g) {
  // 自动选 A
  applyTrumpSelection(g, g.isFirstRound ? 2 : (g.dealerTeam === 1 ? g.redLevel : g.blueLevel));
}

function applyTrumpSelection(g, rank) {
  stopCountdown(g);
  g.trumpRank = rank;
  g.phase = GamePhase.PLAYING;
  g.firstPlay = true;
  g.currentPlayer = 0; // 第一盘从0号位开始
  if (g.lastWinnerId !== null) g.currentPlayer = g.lastWinnerId;

  // 用将牌信息重新排序手牌
  for (const p of g.players) {
    p.sortedCards = sortCards(p.hand, g.jokers, rank);
  }

  addMessage(`将牌设定为：${rank === 14 ? 'A' : rank === 13 ? 'K' : rank === 12 ? 'Q' : rank === 11 ? 'J' : rank}，游戏正式开始！`, 'success');
  broadcastRoomState();
  broadcastAllHands();

  // 如果当前玩家是 AI（没有对应 client），则触发 AI 出牌
  scheduleAIPlayIfNeeded(g);
  startPlayCountdown(g);
}

// ─── 出牌逻辑 ─────────────────────────────────────────
function handlePlayCards(clientId, msg) {
  const info = clientInfo.get(clientId);
  if (!info || !room.game) return;
  const g = room.game;
  if (g.phase !== GamePhase.PLAYING) return;
  if (info.seatIndex !== g.currentPlayer) {
    sendTo(clientId, { type: 'error', message: '还没到你出牌' });
    return;
  }
  const cardIds = msg.cardIds;
  if (!Array.isArray(cardIds) || cardIds.length === 0) return;

  const player = g.players[info.seatIndex];
  const cards = player.hand.filter(c => cardIds.includes(c.id));
  if (cards.length !== cardIds.length) {
    sendTo(clientId, { type: 'error', message: '手牌中没有该牌' });
    return;
  }

  const result = tryPlayCards(g, info.seatIndex, cards);
  if (!result.ok) {
    sendTo(clientId, { type: 'error', message: result.reason });
    return;
  }
}

function handlePassPlay(clientId) {
  const info = clientInfo.get(clientId);
  if (!info || !room.game) return;
  const g = room.game;
  if (g.phase !== GamePhase.PLAYING) return;
  if (info.seatIndex !== g.currentPlayer) return;
  if (g.firstPlay) {
    sendTo(clientId, { type: 'error', message: '首发不能不出' });
    return;
  }
  
  // 检查是否轮到最后出牌者（其他人都pass了，必须出牌）
  const seatIndex = info.seatIndex;
  
  // 找到最后一个有效出牌（非pass）的玩家
  const lastValidPlay = getLastValidPlay(g);
  if (lastValidPlay && lastValidPlay.seatIndex === seatIndex) {
    // 当前玩家是最后一个出牌者，检查其他活跃玩家是否都已pass
    const activePlayers = g.players.filter(p => !p.isOut);
    const otherActivePlayers = activePlayers.filter(p => p.seatIndex !== seatIndex);
    
    // 检查从最后一个有效出牌之后，其他活跃玩家是否都pass了
    let lastValidPlayIdx = -1;
    for (let i = g.playedCards.length - 1; i >= 0; i--) {
      if (g.playedCards[i].cards && g.playedCards[i].cards.length > 0) {
        lastValidPlayIdx = i;
        break;
      }
    }
    const playsAfterLastValid = g.playedCards.slice(lastValidPlayIdx + 1);
    
    // 其他活跃玩家是否都在最后一个有效出牌之后pass了
    const allOthersPassed = otherActivePlayers.every(p => 
      playsAfterLastValid.some(r => r.seatIndex === p.seatIndex && r.cards.length === 0)
    );
    
    if (allOthersPassed && otherActivePlayers.length > 0) {
      // 其他活跃玩家都已pass，必须出牌
      sendTo(clientId, { type: 'error', message: '其他人都已不出，你必须出牌' });
      return;
    }
  }
  
  doPass(g, info.seatIndex);
}

function tryPlayCards(g, seatIndex, cards) {
  const cardType = recognizeCardType(cards, g.jokers, g.trumpRank);
  if (!cardType) return { ok: false, reason: '无效牌型' };

  if (!g.firstPlay && g.currentRoundCardCount !== null && cards.length !== g.currentRoundCardCount) {
    return { ok: false, reason: `本轮需出${g.currentRoundCardCount}张牌` };
  }

  // 非首发：必须比当前最大牌大
  if (!g.firstPlay) {
    const lastValidPlay = getLastValidPlay(g);
    if (lastValidPlay && lastValidPlay.cards && lastValidPlay.cards.length > 0) {
      const cmp = compareCards(lastValidPlay.cards, cards, g.jokers, g.trumpRank);
      if (cmp !== 1) {
        return { ok: false, reason: '出的牌不够大' };
      }
    }
  }

  stopCountdown(g);

  // 执行出牌
  const player = g.players[seatIndex];
  player.hand = player.hand.filter(c => !cards.find(pc => pc.id === c.id));
  player.sortedCards = sortCards(player.hand, g.jokers, g.trumpRank);
  player.cardCount = player.hand.length;

  if (g.firstPlay) {
    g.currentRoundCardCount = cards.length;
    g.roundStarter = seatIndex;
    g.firstPlay = false;
  }

  g.playedCards.push({ seatIndex, cards });
  g.playerShowCards[seatIndex] = cards;

  addMessage(`${player.name} 出牌：${cards.map(getCardDisplay).join(' ')}`, 'info');

  // 检查是否出完牌
  if (player.hand.length === 0) {
    player.isOut = true;
    g.finishOrder.push(seatIndex);
    addMessage(`${player.name} 出完牌！${['头游', '二游', '三游', '四游', '五游', '末游'][g.finishOrder.length - 1]}`, 'success');
  }

  // 检查是否一方三名选手全部出完 - 立即结束
  const redTeamFinished = g.players.filter(p => p.team === 1 && p.isOut).length === 3;
  const blueTeamFinished = g.players.filter(p => p.team === 2 && p.isOut).length === 3;
  
  if (redTeamFinished || blueTeamFinished) {
    // 找出最后出完牌的那方选手（作为逆时针起点）
    const winningTeam = redTeamFinished ? 1 : 2;
    const lastFinishedInWinningTeam = [...g.finishOrder].reverse().find(id => g.players[id].team === winningTeam);
    
    // 找出未出完的对方选手，按逆时针（座位号减1循环）排序
    const losingTeam = winningTeam === 1 ? 2 : 1;
    const remainingLosers = g.players.filter(p => p.team === losingTeam && !p.isOut);
    
    if (remainingLosers.length > 0 && lastFinishedInWinningTeam !== undefined) {
      // 从最后出完的选手开始，逆时针方向找到对方选手并分配排位
      // 逆时针 = 座位号减1（循环）
      let currentSeat = lastFinishedInWinningTeam;
      const assignedLosers = [];
      
      // 最多循环6次，避免死循环
      for (let i = 0; i < 6 && assignedLosers.length < remainingLosers.length; i++) {
        currentSeat = (currentSeat - 1 + 6) % 6; // 逆时针
        const player = g.players[currentSeat];
        if (player.team === losingTeam && !player.isOut) {
          assignedLosers.push(player);
          player.isOut = true;
          g.finishOrder.push(currentSeat);
        }
      }
      
      // 添加消息提示
      addMessage(`${winningTeam === 1 ? '红方' : '蓝方'}全部出完！游戏结束`, 'success');
    }
    
    setTimeout(() => handleGameEnd(g), 500);
    broadcastRoomState();
    broadcastAllHands();
    return { ok: true };
  }

  // 检查游戏是否结束（5人出完）
  if (g.finishOrder.length >= 5) {
    const remaining = g.players.filter(p => !p.isOut);
    if (remaining.length === 1) {
      g.finishOrder.push(remaining[0].seatIndex);
      remaining[0].isOut = true;
    }
  }
  if (g.finishOrder.length >= 6) {
    setTimeout(() => handleGameEnd(g), 500);
    broadcastRoomState();
    broadcastAllHands();
    return { ok: true };
  }

  // 检查本轮是否结束（其他人都不出了）
  checkRoundEnd(g, seatIndex);

  broadcastRoomState();
  broadcastAllHands();
  return { ok: true };
}

function doPass(g, seatIndex) {
  stopCountdown(g);
  const player = g.players[seatIndex];
  g.playedCards.push({ seatIndex, cards: [] });
  player.hasAsked = true;
  addMessage(`${player.name} 不出`, 'info');
  checkRoundEnd(g, seatIndex);
  broadcastRoomState();
}

function getLastValidPlay(g) {
  // 找到本轮最后一次有效出牌（非pass），返回完整记录
  for (let i = g.playedCards.length - 1; i >= 0; i--) {
    const rec = g.playedCards[i];
    if (rec.cards && rec.cards.length > 0) {
      return rec; // 返回完整记录 { seatIndex, cards }
    }
  }
  return null;
}

function checkRoundEnd(g, lastPlaySeatIndex) {
  // 检查是否一轮结束：从最后一个有效出牌者出发，其他人都pass了
  const activePlayers = g.players.filter(p => !p.isOut);
  if (activePlayers.length <= 1) {
    // 只剩一人，进入下一阶段
    nextTurn(g, lastPlaySeatIndex);
    return;
  }

  // 找到最后一个有效出牌者
  const lastValidPlay = getLastValidPlay(g);
  if (!lastValidPlay) {
    // 没有有效出牌，继续下一个玩家
    nextTurn(g, lastPlaySeatIndex);
    return;
  }

  // 找到最后一个有效出牌的索引
  let lastValidPlayIdx = -1;
  for (let i = g.playedCards.length - 1; i >= 0; i--) {
    if (g.playedCards[i].cards && g.playedCards[i].cards.length > 0) {
      lastValidPlayIdx = i;
      break;
    }
  }
  
  // 获取最后一个有效出牌之后的所有出牌记录
  const playsAfterLastValid = g.playedCards.slice(lastValidPlayIdx + 1);
  
  // 检查其他活跃玩家是否都在最后一个有效出牌之后pass了
  const otherActivePlayers = activePlayers.filter(p => p.seatIndex !== lastValidPlay.seatIndex);
  
  if (otherActivePlayers.length === 0) {
    // 只剩最后一个出牌者，一轮结束
    endRoundWithStarter(g, lastValidPlay.seatIndex);
    return;
  }
  
  const allOthersPassed = otherActivePlayers.every(p => 
    playsAfterLastValid.some(r => r.seatIndex === p.seatIndex && r.cards.length === 0)
  );

  if (allOthersPassed) {
    // 所有人都 pass 了，本轮结束，最后一个出牌者继续首发
    endRoundWithStarter(g, lastValidPlay.seatIndex);
    return;
  }

  nextTurn(g, lastPlaySeatIndex);
}

function getRoundPlays(g) {
  // 获取当前轮次的出牌（从最后一次 firstPlay 开始）
  // 简化：返回最近 SEAT_COUNT 条记录
  return g.playedCards.slice(-SEAT_COUNT);
}

function endRoundWithStarter(g, starterSeatIndex) {
  // 清除本轮出牌记录和桌面牌显示
  g.playedCards = [];
  g.playerShowCards = {};
  g.firstPlay = true;
  g.currentRoundCardCount = null;
  // 指定的玩家首发
  if (g.players[starterSeatIndex] && !g.players[starterSeatIndex].isOut) {
    g.currentPlayer = starterSeatIndex;
  } else {
    g.currentPlayer = findNextActive(g, starterSeatIndex);
  }
  g.roundStarter = g.currentPlayer;
  // 重置所有 hasAsked
  for (const p of g.players) p.hasAsked = false;

  broadcastRoomState();
  scheduleAIPlayIfNeeded(g);
  startPlayCountdown(g);
}

function endRound(g) {
  // 清除本轮出牌记录和桌面牌显示
  g.playedCards = [];
  g.playerShowCards = {};
  g.firstPlay = true;
  g.currentRoundCardCount = null;
  // roundStarter 重新首发
  const starter = g.roundStarter;
  if (g.players[starter] && !g.players[starter].isOut) {
    g.currentPlayer = starter;
  } else {
    g.currentPlayer = findNextActive(g, starter);
  }
  g.roundStarter = g.currentPlayer;
  // 重置所有 hasAsked
  for (const p of g.players) p.hasAsked = false;

  broadcastRoomState();
  scheduleAIPlayIfNeeded(g);
  startPlayCountdown(g);
}

function nextTurn(g, lastSeatIndex) {
  const next = findNextActive(g, lastSeatIndex);
  g.currentPlayer = next;
  // 重置所有 hasAsked
  for (const p of g.players) p.hasAsked = false;
  scheduleAIPlayIfNeeded(g);
  startPlayCountdown(g);
}

function findNextActive(g, fromSeat) {
  let next = (fromSeat + 1) % SEAT_COUNT;
  let count = 0;
  while (g.players[next]?.isOut && count < SEAT_COUNT) {
    next = (next + 1) % SEAT_COUNT;
    count++;
  }
  return next;
}

// ─── AI 出牌 ─────────────────────────────────────────
function isAISeat(g, seatIndex) {
  // 没有对应在线客户端的座位视为 AI
  const player = g.players[seatIndex];
  if (!player) return false;
  const cid = player.clientId;
  // 检查是否有活跃连接
  return !cid || !clients.has(cid) || clients.get(cid).readyState !== WebSocket.OPEN;
}

function scheduleAIPlayIfNeeded(g) {
  if (g.phase !== GamePhase.PLAYING) return;
  const seatIndex = g.currentPlayer;
  if (!isAISeat(g, seatIndex)) return;

  // AI 延迟 1~2 秒出牌
  const delay = 1000 + Math.random() * 1000;
  setTimeout(async () => {
    if (g.phase !== GamePhase.PLAYING) return;
    if (g.currentPlayer !== seatIndex) return;
    await doAIPlay(g, seatIndex);
  }, delay);
}

async function doAIPlay(g, seatIndex) {
  const player = g.players[seatIndex];
  if (!player || player.isOut) return;

  const lastValidPlay = getLastValidPlay(g);

  // ── 优先调用推理服务（训练好的神经网络）──
  let cards = null;
  if (AI_INFERENCE_URL) {
    try {
      const body = JSON.stringify({
        seatIndex,
        hand:          player.hand,
        currentPlayed: lastValidPlay ? lastValidPlay.cards : [],
        playedAll:     g.playedCards.flatMap(p => p.cards),
        handSizes:     g.players.map(p => p ? p.hand.length : 0),
        trumpRank:     g.trumpRank,
        isFirstPlay:   g.firstPlay,
      });
      const res = await fetch(AI_INFERENCE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        signal: AbortSignal.timeout(2000),  // 2秒超时
      });
      if (res.ok) {
        const json = await res.json();
        if (!json.pass && Array.isArray(json.action) && json.action.length > 0) {
          // 把推理服务返回的牌 id 匹配回 player.hand（保证引用正确）
          const actionIds = new Set(json.action.map(c => c.id));
          cards = player.hand.filter(c => actionIds.has(c.id));
          if (cards.length !== json.action.length) cards = null; // id不匹配，降级
        } else {
          cards = [];  // pass
        }
      }
    } catch (e) {
      // 推理服务不可用，静默降级
    }
  }

  // ── 降级：使用规则AI ──
  if (cards === null) {
    cards = aiSelectCards(
      player.hand,
      g.jokers,
      lastValidPlay ? lastValidPlay.cards : null,
      g.firstPlay,
      g.currentRoundCardCount,
      g.trumpRank
    );
  }

  if (!cards || cards.length === 0) {
    if (!g.firstPlay) {
      doPass(g, seatIndex);
    }
    return;
  }

  tryPlayCards(g, seatIndex, cards);
}

// 推理服务地址（启动时可通过环境变量 AI_INFERENCE_URL 指定）
const AI_INFERENCE_URL = process.env.AI_INFERENCE_URL || 'http://localhost:5001/ai_action';

// ─── 游戏结束 / 下一盘 ─────────────────────────────────
function handleGameEnd(g) {
  if (g.phase === GamePhase.ROUND_END || g.phase === GamePhase.TRIBUTE) return;
  stopCountdown(g);

  const order = g.finishOrder;
  const { winnerTeam, score } = calculateScore(order, g.players);

  if (winnerTeam) {
    // 判断胜方是否是庄家队
    const winnerIsDealer = (winnerTeam === g.dealerTeam);
    
    if (winnerIsDealer) {
      // 庄家队赢了
      if (score > 0) {
        // 庄家队赢了并得分 → 庄家队升级
        if (g.dealerTeam === 1) {
          g.scores.red += score;
          g.redLevel = Math.min(g.redLevel + 1, 14);
        } else {
          g.scores.blue += score;
          g.blueLevel = Math.min(g.blueLevel + 1, 14);
        }
        addMessage(`${winnerTeam === 1 ? '红队' : '蓝队'}（庄家）获胜！得${score}分，升级！`, 'success');
      } else {
        // 庄家队赢了但没得分 → 庄家不变，不升级
        addMessage(`${winnerTeam === 1 ? '红队' : '蓝队'}（庄家）头游，但不得分，不升级`, 'info');
      }
      // 庄家队赢了，庄家不变
    } else {
      // 庄家队输了 → 对方成为新庄家，但不升级
      g.dealerTeam = winnerTeam;
      if (score > 0) {
        if (winnerTeam === 1) {
          g.scores.red += score;
        } else {
          g.scores.blue += score;
        }
      }
      addMessage(`${winnerTeam === 1 ? '红队' : '蓝队'}获胜！成为新庄家，但不升级`, 'success');
    }
  }

  g.lastWinnerId = order[0];
  g.roundWinner = order[0];

  const tributeData = checkAndInitTribute(order, g.players);
  if (tributeData) {
    g.pendingTributeInfo = tributeData;
    const names = tributeData.tributors.map(i => g.players[i]?.name).join('、');
    addMessage(`本局结束！${names} 需要在下一盘进贡`, 'info');
  }

  g.phase = GamePhase.ROUND_END;
  broadcastRoomState();

  // 如果所有座位都是 AI（没有真人），自动延迟 5 秒开始下一盘
  const hasHumanPlayer = room.seats.some(s => s && !s.isAI);
  if (!hasHumanPlayer) {
    setTimeout(() => {
      if (room.game && room.game.phase === GamePhase.ROUND_END) {
        startNextRound(room.game);
      }
    }, 5000);
  }
}

function calculateScore(order, players) {
  if (order.length < 6) return { winnerTeam: null, score: 0 };
  const firstTeam = players.find(p => p.seatIndex === order[0])?.team;
  if (!firstTeam) return { winnerTeam: null, score: 0 };
  const opponentTeam = firstTeam === 1 ? 2 : 1;
  let count = 0;
  for (let i = 5; i >= 0; i--) {
    const pTeam = players.find(p => p.seatIndex === order[i])?.team;
    if (pTeam === opponentTeam) count++;
    else break;
  }
  return { winnerTeam: firstTeam, score: count };
}

function checkAndInitTribute(order, players) {
  if (order.length < 6) return null;
  const firstTeam = players.find(p => p.seatIndex === order[0])?.team;
  if (!firstTeam) return null;
  const loserTeam = firstTeam === 1 ? 2 : 1;
  const tributors = [];
  for (let i = 5; i >= 0; i--) {
    const p = players.find(pp => pp.seatIndex === order[i]);
    if (p?.team === loserTeam) tributors.push(order[i]);
    else break;
  }
  if (tributors.length === 0) return null;
  const receivers = [];
  for (let i = 0; i < tributors.length; i++) {
    const p = players.find(pp => pp.seatIndex === order[i]);
    if (p?.team === firstTeam) receivers.push(order[i]);
    else break;
  }
  const count = Math.min(tributors.length, receivers.length);
  return {
    tributors: tributors.slice(0, count),
    receivers: receivers.slice(0, count),
    currentTributorIndex: 0,
    currentReceiverIndex: 0,
    tributes: [],
    returns: [],
    canResist: false,
    resistedPlayers: [],
    jokerCards: [],
    returnSubPhase: 'selecting_candidates',
    returnCandidates: []
  };
}

function handleStartNextRound(clientId) {
  const info = clientInfo.get(clientId);
  if (!info || !room.game) return;
  const g = room.game;
  if (g.phase !== GamePhase.ROUND_END) return;
  // 任何玩家都可以触发下一盘
  startNextRound(g);
}

function startNextRound(g) {
  stopCountdown(g);
  g.phase = GamePhase.DEALING;

  // 直接设定将牌：根据庄家队的级别
  const trumpRank = g.dealerTeam === 1 ? g.redLevel : g.blueLevel;
  g.trumpRank = trumpRank;

  // 重新发牌
  const deck = shuffleDeck(createDeck());
  const hands = dealCards(deck, SEAT_COUNT);
  const allJokers = deck.filter(c => c.isJoker);
  g.jokers = allJokers;

  for (let i = 0; i < SEAT_COUNT; i++) {
    g.players[i].hand = hands[i];
    g.players[i].sortedCards = sortCards(hands[i], allJokers, trumpRank);
    g.players[i].cardCount = hands[i].length;
    g.players[i].isOut = false;
    g.players[i].hasAsked = false;
  }

  g.playedCards = [];
  g.playerShowCards = {};
  g.finishOrder = [];
  g.firstPlay = true;
  g.currentRoundCardCount = null;
  g.roundStarter = null;
  g.roundWinner = null;
  g.tributeInfo = null;
  g.lastTributeInfo = [];
  g.selectedReturnCandidates = [];
  g.isFirstRound = false;

  // 检查是否需要进贡
  if (g.pendingTributeInfo) {
    g.tributeInfo = g.pendingTributeInfo;
    g.pendingTributeInfo = null;
    g.phase = GamePhase.TRIBUTE;
    addMessage('开始进贡阶段...', 'info');
    broadcastRoomState();
    broadcastAllHands();
    scheduleTributeCountdown(g);
  } else {
    // 直接开始游戏
    g.phase = GamePhase.PLAYING;
    g.currentPlayer = g.lastWinnerId !== null ? g.lastWinnerId : 0;
    
    const trumpName = trumpRank === 14 ? 'A' : trumpRank === 13 ? 'K' : trumpRank === 12 ? 'Q' : trumpRank === 11 ? 'J' : String(trumpRank);
    addMessage(`新一局开始！将牌：${trumpName}`, 'success');
    
    broadcastRoomState();
    broadcastAllHands();
    scheduleAIPlayIfNeeded(g);
    startPlayCountdown(g);
  }
}

// ─── 进贡逻辑 ─────────────────────────────────────────
function scheduleTributeCountdown(g) {
  stopCountdown(g);
  if (!g.tributeInfo) return;
  g.tributeCountdown = 15;
  broadcastRoomState();

  g.countdownTimer = setInterval(() => {
    g.tributeCountdown--;
    if (g.tributeCountdown <= 0) {
      stopCountdown(g);
      // 自动进贡：出最大的牌
      autoTribute(g);
    } else {
      broadcastRoomState();
    }
  }, 1000);
}

function autoTribute(g) {
  if (!g.tributeInfo || g.phase !== GamePhase.TRIBUTE) return;
  const ti = g.tributeInfo;
  const tributorSeat = ti.tributors[ti.currentTributorIndex];
  const player = g.players[tributorSeat];
  if (!player) return;

  // 检查抗贡条件
  const jokerCount = countJokersInHand(player.hand);
  if (jokerCount >= 3) {
    // 自动抗贡时，展示所有王
    const jokerCards = getJokerCards(player.hand);
    doResist(g, tributorSeat, jokerCards);
    return;
  }

  const bestCard = getBestCard(player.hand, g.jokers);
  if (bestCard) doTribute(g, tributorSeat, bestCard);
}

function handleTributeResist(clientId, msg) {
  const info = clientInfo.get(clientId);
  if (!info || !room.game) return;
  const g = room.game;
  if (g.phase !== GamePhase.TRIBUTE || !g.tributeInfo) return;
  const ti = g.tributeInfo;
  const currentTributorSeat = ti.tributors[ti.currentTributorIndex];
  if (info.seatIndex !== currentTributorSeat) {
    sendTo(clientId, { type: 'error', message: '不是你进贡' });
    return;
  }
  const jokerIds = msg.jokerIds || [];
  if (jokerIds.length < 3) {
    sendTo(clientId, { type: 'error', message: '抗贡需要展示至少3个王' });
    return;
  }
  // 验证选择的牌都是王
  const player = g.players[info.seatIndex];
  const selectedJokers = jokerIds.map(id => player.hand.find(c => c.id === id)).filter(c => c && c.isJoker);
  if (selectedJokers.length !== jokerIds.length) {
    sendTo(clientId, { type: 'error', message: '选择的牌必须都是王' });
    return;
  }
  doResist(g, info.seatIndex, selectedJokers);
}

function doResist(g, seatIndex, jokerCards) {
  const ti = g.tributeInfo;
  if (!ti) return;
  stopCountdown(g);
  const player = g.players[seatIndex];
  ti.resistedPlayers.push(seatIndex);
  ti.jokerCards = jokerCards;
  ti.currentTributorIndex++;
  ti.currentReceiverIndex++;
  addMessage(`${player.name} 抗贡！展示 ${jokerCards.length} 个王`, 'success');

  if (ti.currentTributorIndex >= ti.tributors.length) {
    setTimeout(() => proceedToReturnOrEnd(g), 1500);
  } else {
    broadcastRoomState();
    scheduleTributeCountdown(g);
  }
}

function handleTributeConfirm(clientId, msg) {
  const info = clientInfo.get(clientId);
  if (!info || !room.game) return;
  const g = room.game;
  if (g.phase !== GamePhase.TRIBUTE || !g.tributeInfo) return;
  const ti = g.tributeInfo;
  const currentTributorSeat = ti.tributors[ti.currentTributorIndex];
  if (info.seatIndex !== currentTributorSeat) {
    sendTo(clientId, { type: 'error', message: '不是你进贡' });
    return;
  }
  const cardId = msg.cardId;
  const player = g.players[info.seatIndex];
  const card = player.hand.find(c => c.id === cardId);
  if (!card) {
    sendTo(clientId, { type: 'error', message: '手牌中没有该牌' });
    return;
  }
  doTribute(g, info.seatIndex, card);
}

function doTribute(g, seatIndex, card) {
  const ti = g.tributeInfo;
  if (!ti) return;
  stopCountdown(g);
  const player = g.players[seatIndex];
  const receiverSeat = ti.receivers[ti.currentTributorIndex];

  player.hand = player.hand.filter(c => c.id !== card.id);
  player.sortedCards = sortCards(player.hand, g.jokers, g.trumpRank);
  player.cardCount = player.hand.length;

  ti.tributes.push({ tributorId: seatIndex, card, receiverId: receiverSeat });
  ti.currentTributorIndex++;
  ti.currentReceiverIndex++;

  addMessage(`${player.name} 进贡 ${getCardDisplay(card)}`, 'info');

  if (ti.currentTributorIndex >= ti.tributors.length) {
    setTimeout(() => proceedToReturnOrEnd(g), 1000);
  } else {
    broadcastRoomState();
    broadcastAllHands();
    scheduleTributeCountdown(g);
  }
}

function proceedToReturnOrEnd(g) {
  const ti = g.tributeInfo;
  if (!ti) return;

  const validTributes = ti.tributes.filter(t => !ti.resistedPlayers.includes(t.tributorId));
  if (validTributes.length > 0) {
    ti.currentTributorIndex = 0;
    ti.currentReceiverIndex = 0;
    ti.returnSubPhase = 'selecting_candidates';
    ti.returnCandidates = [];
    g.phase = GamePhase.TRIBUTE_RETURN;
    g.returnCountdown = 15;
    addMessage('进贡完成，开始还贡...', 'info');
    broadcastRoomState();
    broadcastAllHands();
    scheduleReturnCountdown(g);
  } else {
    finishTributePhase(g);
  }
}

function scheduleReturnCountdown(g) {
  stopCountdown(g);
  g.returnCountdown = 15;

  g.countdownTimer = setInterval(() => {
    g.returnCountdown--;
    if (g.returnCountdown <= 0) {
      stopCountdown(g);
      autoReturn(g);
    } else {
      broadcastRoomState();
    }
  }, 1000);
}

function autoReturn(g) {
  const ti = g.tributeInfo;
  if (!ti || g.phase !== GamePhase.TRIBUTE_RETURN) return;
  const validTributes = ti.tributes.filter(t => !ti.resistedPlayers.includes(t.tributorId));
  const currentTribute = validTributes[ti.currentTributorIndex];
  if (!currentTribute) return;

  if (ti.returnSubPhase === 'selecting_candidates') {
    // 受贡方自动选候选牌
    const receiverSeat = currentTribute.receiverId;
    const receiver = g.players[receiverSeat];
    const needed = getReturnCandidateCount(currentTribute.card);
    const sorted = sortCards(receiver.hand, g.jokers, g.trumpRank);
    const candidates = sorted.slice(-needed); // 选最小的牌
    doReturnCandidatesConfirm(g, receiverSeat, candidates);
  } else {
    // 进贡方自动挑第一张
    const tributorSeat = currentTribute.tributorId;
    doReturnConfirm(g, tributorSeat, ti.returnCandidates[0]);
  }
}

function getReturnCandidateCount(tributeCard) {
  if (tributeCard.isBigJoker) return 3;
  if (tributeCard.isSmallJoker) return 2;
  return 1;
}

function handleReturnCandidatesConfirm(clientId, msg) {
  const info = clientInfo.get(clientId);
  if (!info || !room.game) return;
  const g = room.game;
  if (g.phase !== GamePhase.TRIBUTE_RETURN || !g.tributeInfo) return;
  const ti = g.tributeInfo;
  const validTributes = ti.tributes.filter(t => !ti.resistedPlayers.includes(t.tributorId));
  const currentTribute = validTributes[ti.currentTributorIndex];
  if (!currentTribute) return;
  if (info.seatIndex !== currentTribute.receiverId) {
    sendTo(clientId, { type: 'error', message: '不是你还贡' });
    return;
  }
  const cardIds = msg.cardIds;
  const needed = getReturnCandidateCount(currentTribute.card);
  if (!Array.isArray(cardIds) || cardIds.length !== needed) {
    sendTo(clientId, { type: 'error', message: `需要选 ${needed} 张牌` });
    return;
  }
  const receiver = g.players[info.seatIndex];
  const cards = cardIds.map(id => receiver.hand.find(c => c.id === id)).filter(Boolean);
  if (cards.length !== needed) {
    sendTo(clientId, { type: 'error', message: '手牌中没有该牌' });
    return;
  }
  doReturnCandidatesConfirm(g, info.seatIndex, cards);
}

function doReturnCandidatesConfirm(g, receiverSeat, cards) {
  const ti = g.tributeInfo;
  if (!ti) return;
  stopCountdown(g);
  const needed = getReturnCandidateCount(
    ti.tributes.filter(t => !ti.resistedPlayers.includes(t.tributorId))[ti.currentTributorIndex]?.card
  );

  if (needed === 1) {
    // 直接还贡
    const validTributes = ti.tributes.filter(t => !ti.resistedPlayers.includes(t.tributorId));
    const currentTribute = validTributes[ti.currentTributorIndex];
    applyReturn(g, currentTribute, cards[0]);
    checkReturnEnd(g);
  } else {
    // 等进贡方挑牌
    ti.returnSubPhase = 'tributor_picking';
    ti.returnCandidates = cards;
    g.returnCountdown = 15;
    broadcastRoomState();
    scheduleReturnCountdown(g);
  }
}

function handleReturnConfirm(clientId, msg) {
  const info = clientInfo.get(clientId);
  if (!info || !room.game) return;
  const g = room.game;
  if (g.phase !== GamePhase.TRIBUTE_RETURN || !g.tributeInfo) return;
  const ti = g.tributeInfo;
  const validTributes = ti.tributes.filter(t => !ti.resistedPlayers.includes(t.tributorId));
  const currentTribute = validTributes[ti.currentTributorIndex];
  if (!currentTribute) return;
  if (info.seatIndex !== currentTribute.tributorId) {
    sendTo(clientId, { type: 'error', message: '不是你选牌' });
    return;
  }
  const cardId = msg.cardId;
  const card = ti.returnCandidates.find(c => c.id === cardId);
  if (!card) {
    sendTo(clientId, { type: 'error', message: '候选牌中没有该牌' });
    return;
  }
  doReturnConfirm(g, info.seatIndex, card);
}

function doReturnConfirm(g, tributorSeat, card) {
  const ti = g.tributeInfo;
  if (!ti) return;
  stopCountdown(g);
  const validTributes = ti.tributes.filter(t => !ti.resistedPlayers.includes(t.tributorId));
  const currentTribute = validTributes[ti.currentTributorIndex];
  if (!currentTribute) return;
  applyReturn(g, currentTribute, card);
  checkReturnEnd(g);
}

function applyReturn(g, currentTribute, returnCard) {
  const { tributorId, receiverId, card: tributeCard } = currentTribute;
  const receiver = g.players[receiverId];
  const tributor = g.players[tributorId];

  // 受贡方移除还贡牌，收进贡牌
  receiver.hand = receiver.hand.filter(c => c.id !== returnCard.id);
  receiver.hand.push(tributeCard);
  receiver.sortedCards = sortCards(receiver.hand, g.jokers, g.trumpRank);
  receiver.cardCount = receiver.hand.length;

  // 进贡方收到还贡牌
  tributor.hand.push(returnCard);
  tributor.sortedCards = sortCards(tributor.hand, g.jokers, g.trumpRank);
  tributor.cardCount = tributor.hand.length;

  const ti = g.tributeInfo;
  ti.returns.push({ tributorId, card: returnCard, receiverId });
  ti.currentTributorIndex++;
  ti.returnSubPhase = 'selecting_candidates';
  ti.returnCandidates = [];

  addMessage(`${receiver.name} 还贡 ${getCardDisplay(returnCard)} 给 ${tributor.name}`, 'info');
  broadcastRoomState();
  broadcastAllHands();
}

function checkReturnEnd(g) {
  const ti = g.tributeInfo;
  if (!ti) return;
  const validTributes = ti.tributes.filter(t => !ti.resistedPlayers.includes(t.tributorId));
  if (ti.currentTributorIndex >= validTributes.length) {
    setTimeout(() => finishTributePhase(g), 1000);
  } else {
    g.returnCountdown = 15;
    broadcastRoomState();
    scheduleReturnCountdown(g);
  }
}

function finishTributePhase(g) {
  stopCountdown(g);
  const ti = g.tributeInfo;
  if (ti) {
    g.lastTributeInfo = ti.tributes
      .filter(t => !ti.resistedPlayers.includes(t.tributorId))
      .map(t => ({
        tributorName: g.players[t.tributorId]?.name,
        receiverName: g.players[t.receiverId]?.name,
        card: t.card
      }));
  }
  g.tributeInfo = null;

  // 进贡结束，直接开始游戏（将牌已在发牌时设定）
  g.phase = GamePhase.PLAYING;
  g.currentPlayer = g.lastWinnerId !== null ? g.lastWinnerId : 0;
  
  const trumpRank = g.trumpRank;
  const trumpName = trumpRank === 14 ? 'A' : trumpRank === 13 ? 'K' : trumpRank === 12 ? 'Q' : trumpRank === 11 ? 'J' : String(trumpRank);
  addMessage(`进贡结束，游戏开始！将牌：${trumpName}`, 'success');
  
  broadcastRoomState();
  broadcastAllHands();
  scheduleAIPlayIfNeeded(g);
  startPlayCountdown(g);
}

// ─── 倒计时管理 ─────────────────────────────────────────
function startPlayCountdown(g) {
  stopCountdown(g);
  if (g.phase !== GamePhase.PLAYING) return;
  // 每个人出牌等待时间统一为 60 秒
  g.countdown = 60;

  g.countdownTimer = setInterval(() => {
    g.countdown--;
    if (g.countdown <= 0) {
      stopCountdown(g);
      // 超时：自动出牌或pass
      if (isAISeat(g, g.currentPlayer)) return; // AI 已经有自己的逻辑
      const seatIndex = g.currentPlayer;
      if (g.firstPlay) {
        // 强制出最小的牌
        const player = g.players[seatIndex];
        const sorted = sortCards(player.hand, g.jokers, g.trumpRank);
        if (sorted.length > 0) {
          tryPlayCards(g, seatIndex, [sorted[sorted.length - 1]]);
        }
      } else {
        doPass(g, seatIndex);
      }
    } else {
      broadcastRoomState();
    }
  }, 1000);
}

function stopCountdown(g) {
  if (g.countdownTimer) {
    clearInterval(g.countdownTimer);
    g.countdownTimer = null;
  }
}

function startCountdown(g, type, seconds, onEnd) {
  stopCountdown(g);
  g.countdown = seconds;
  broadcastRoomState();
  g.countdownTimer = setInterval(() => {
    g.countdown--;
    if (g.countdown <= 0) {
      stopCountdown(g);
      onEnd();
    } else {
      broadcastRoomState();
    }
  }, 1000);
}

console.log('服务器模块已加载');
