/**
 * GameContext.tsx
 * WebSocket 连接层 + 全局游戏状态管理
 * 所有组件通过 useGame() hook 访问游戏状态和发送操作
 */

import { createContext, useContext, useEffect, useRef, useState, useCallback, ReactNode } from 'react';
import { Card } from './types';

// ─── 类型定义 ─────────────────────────────────────────
export interface SeatInfo {
  seatIndex: number;
  label: string;
  team: number; // 1=红队, 2=蓝队
  occupied: boolean;
  clientId: string | null;
  nickname: string | null;
  ready: boolean;
  disconnected?: boolean;
  playerId?: string;
  isAI?: boolean;
  aiId?: string | null;
}

export interface PlayerPublic {
  seatIndex: number;
  name: string;
  team: number;
  isOut: boolean;
  cardCount: number;
  hasAsked: boolean;
  disconnected?: boolean;  // 是否断线中
}

export interface TributeInfoPublic {
  tributors: number[];
  receivers: number[];
  currentTributorIndex: number;
  currentReceiverIndex: number;
  tributes: { tributorId: number; card: Card; receiverId: number }[];
  returns: { tributorId: number; card: Card; receiverId: number }[];
  canResist: boolean;
  resistedPlayers: number[];
  jokerCards: Card[];
  returnSubPhase: 'selecting_candidates' | 'tributor_picking';
  returnCandidates: Card[];
}

export interface GamePublic {
  phase: string;
  players: PlayerPublic[];
  currentPlayer: number; // seatIndex
  playedCards: { seatIndex: number; cards: Card[] }[];
  playerShowCards: Record<number, Card[]>;
  finishedPlayerLastCards: Record<number, Card[]>; // 已出完玩家的最后出牌
  currentRoundCardCount: number | null;
  roundStarter: number | null;
  roundWinner: number | null;
  firstPlay: boolean;
  jokers: Card[];
  trumpRank: number | null;
  finishOrder: number[];
  scores: { red: number; blue: number };
  dealerTeam: number;
  lastWinnerId: number | null;
  isFirstRound: boolean;
  redLevel: number;
  blueLevel: number;
  tributeInfo: TributeInfoPublic | null;
  lastTributeInfo: { tributorName: string; receiverName: string; card: Card }[];
  countdown: number;
  tributeCountdown: number;
  returnCountdown: number;
  tributeForceNoResist: boolean;
  selectedReturnCandidates: Card[];
}

export interface AICharacter {
  id: string;
  name: string;
  emoji: string;
}

export interface RoomSnapshot {
  seats: SeatInfo[];
  game: GamePublic | null;
  messages: { id: number; text: string; type: string; timestamp: number }[];
  aiCharacters: AICharacter[];
  usedAIIds: string[];
}

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected';

export interface GameContextValue {
  // 连接信息
  status: ConnectionStatus;
  clientId: string | null;
  myNickname: string | null;
  mySeatIndex: number; // -1 = 观战
  isSpectator: boolean;

  // 房间状态
  room: RoomSnapshot | null;

  // 我的手牌（只有我自己的）
  myHand: Card[];
  mySortedCards: Card[];

  // 操作函数
  setNickname: (name: string) => void;
  takeSeat: (seatIndex: number) => void;
  leaveSeat: () => void;
  toggleReady: () => void;
  addAI: (seatIndex: number, aiId: string) => void;
  removeAI: (seatIndex: number) => void;
  selectTrump: (rank: number) => void;
  playCards: (cardIds: string[]) => void;
  passPlay: () => void;
  startNextRound: () => void;
  tributeResist: (jokerIds: string[]) => void;
  tributeConfirm: (cardId: string) => void;
  returnCandidatesConfirm: (cardIds: string[]) => void;
  returnConfirm: (cardId: string) => void;

  // 错误消息
  lastError: string | null;
  clearError: () => void;
}

// ─── Context ─────────────────────────────────────────
const GameContext = createContext<GameContextValue | null>(null);

export function useGame(): GameContextValue {
  const ctx = useContext(GameContext);
  if (!ctx) throw new Error('useGame must be used inside GameProvider');
  return ctx;
}

// ─── Provider ─────────────────────────────────────────
const WS_URL = (() => {
  // 优先用环境变量
  if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL;
  // 生产环境：与页面同域
  if (import.meta.env.PROD) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}`;
  }
  // 开发环境
  return `ws://${window.location.hostname}:3002`;
})();

export function GameProvider({ children }: { children: ReactNode }) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [status, setStatus] = useState<ConnectionStatus>('connecting');
  const [clientId, setClientId] = useState<string | null>(null);
  const [myNickname, setMyNickname] = useState<string | null>(null);
  const [mySeatIndex, setMySeatIndex] = useState<number>(-1);
  const [room, setRoom] = useState<RoomSnapshot | null>(null);
  const [myHand, setMyHand] = useState<Card[]>([]);
  const [mySortedCards, setMySortedCards] = useState<Card[]>([]);
  const [lastError, setLastError] = useState<string | null>(null);

  // 当前 clientId 和 seatIndex 供回调使用
  const clientIdRef = useRef<string | null>(null);
  const mySeatRef = useRef<number>(-1);
  
  // playerId 用于断线重连识别
  const playerIdRef = useRef<string | null>(null);
  
  // 获取或生成 playerId
  const getOrCreatePlayerId = useCallback(() => {
    const stored = localStorage.getItem('daguailuzi_player_id');
    if (stored) {
      playerIdRef.current = stored;
      return stored;
    }
    // 生成新的 playerId（UUID 格式）
    const newId = 'player_' + Date.now() + '_' + Math.random().toString(36).substring(2, 11);
    localStorage.setItem('daguailuzi_player_id', newId);
    playerIdRef.current = newId;
    return newId;
  }, []);

  useEffect(() => { clientIdRef.current = clientId; }, [clientId]);
  useEffect(() => { mySeatRef.current = mySeatIndex; }, [mySeatIndex]);

  const send = useCallback((data: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current) {
      try { wsRef.current.close(); } catch (_) {}
    }
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus('connected');
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      // 发送 playerId 用于断线重连识别
      const playerId = getOrCreatePlayerId();
      ws.send(JSON.stringify({ type: 'set_player_id', playerId }));
    };

    ws.onclose = () => {
      setStatus('disconnected');
      // 3秒后重连
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      setStatus('disconnected');
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        switch (msg.type) {
          case 'welcome':
            setClientId(msg.clientId);
            clientIdRef.current = msg.clientId;
            break;

          case 'player_id_ok':
            // playerId 已被服务器记录
            console.log('[重连] playerId 已记录:', msg.playerId);
            break;

          case 'reconnect_success':
            // 重连成功，恢复状态
            console.log('[重连] 重连成功:', msg.nickname, '座位:', msg.seatIndex);
            setMyNickname(msg.nickname);
            setMySeatIndex(msg.seatIndex);
            mySeatRef.current = msg.seatIndex;
            if (msg.hand) {
              setMyHand(msg.hand);
            }
            if (msg.game) {
              // 更新游戏状态
              setRoom(prev => prev ? { ...prev, game: msg.game } : null);
            }
            break;

          case 'room_state':
            setRoom(msg.data);
            // 从房间状态同步自己的 seatIndex（优先用 playerId，其次用 clientId）
            if (msg.data?.seats) {
              let mySeat = null;
              if (playerIdRef.current) {
                mySeat = msg.data.seats.find(
                  (s: SeatInfo & { playerId?: string }) => s.playerId === playerIdRef.current
                );
              }
              if (!mySeat && clientIdRef.current) {
                mySeat = msg.data.seats.find(
                  (s: SeatInfo) => s.clientId === clientIdRef.current
                );
              }
              const newSeat = mySeat ? mySeat.seatIndex : -1;
              setMySeatIndex(newSeat);
              mySeatRef.current = newSeat;
              if (mySeat?.nickname) setMyNickname(mySeat.nickname);
            }
            break;

          case 'your_hand':
            setMyHand(msg.hand || []);
            setMySortedCards(msg.sortedCards || []);
            break;

          case 'nickname_ok':
            setMyNickname(msg.nickname);
            break;

          case 'error':
            setLastError(msg.message);
            break;
          
          case 'game_reset':
            // 游戏被重置，显示原因并刷新页面
            alert(msg.message || '游戏已重置，请重新加入');
            window.location.reload();
            break;
        }
      } catch (e) {
        console.error('WS消息解析失败', e);
      }
    };
  }, [getOrCreatePlayerId]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // ─── 操作函数 ───────────────────────────────────────
  const setNicknameAction = useCallback((name: string) => {
    setMyNickname(name);
    send({ type: 'set_nickname', nickname: name });
  }, [send]);

  const takeSeat = useCallback((seatIndex: number) => {
    send({ type: 'take_seat', seatIndex });
  }, [send]);

  const leaveSeat = useCallback(() => {
    send({ type: 'leave_seat' });
  }, [send]);

  const toggleReady = useCallback(() => {
    send({ type: 'toggle_ready' });
  }, [send]);

  const addAI = useCallback((seatIndex: number, aiId: string) => {
    send({ type: 'add_ai', seatIndex, aiId });
  }, [send]);

  const removeAI = useCallback((seatIndex: number) => {
    send({ type: 'remove_ai', seatIndex });
  }, [send]);

  const selectTrump = useCallback((rank: number) => {
    send({ type: 'select_trump', rank });
  }, [send]);

  const playCards = useCallback((cardIds: string[]) => {
    send({ type: 'play_cards', cardIds });
  }, [send]);

  const passPlay = useCallback(() => {
    send({ type: 'pass_play' });
  }, [send]);

  const startNextRound = useCallback(() => {
    send({ type: 'start_next_round' });
  }, [send]);

  const tributeResist = useCallback((jokerIds: string[]) => {
    send({ type: 'tribute_resist', jokerIds });
  }, [send]);

  const tributeConfirm = useCallback((cardId: string) => {
    send({ type: 'tribute_confirm', cardId });
  }, [send]);

  const returnCandidatesConfirm = useCallback((cardIds: string[]) => {
    send({ type: 'return_candidates_confirm', cardIds });
  }, [send]);

  const returnConfirm = useCallback((cardId: string) => {
    send({ type: 'return_confirm', cardId });
  }, [send]);

  const clearError = useCallback(() => setLastError(null), []);

  const isSpectator = mySeatIndex < 0;

  return (
    <GameContext.Provider value={{
      status, clientId, myNickname, mySeatIndex, isSpectator,
      room, myHand, mySortedCards,
      setNickname: setNicknameAction,
      takeSeat, leaveSeat, toggleReady, addAI, removeAI,
      selectTrump, playCards, passPlay, startNextRound,
      tributeResist, tributeConfirm, returnCandidatesConfirm, returnConfirm,
      lastError, clearError
    }}>
      {children}
    </GameContext.Provider>
  );
}
