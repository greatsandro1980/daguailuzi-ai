/**
 * OnlineGame.tsx
 * 联机版游戏界面 - 从 GameContext 获取状态，操作通过 WebSocket 发送
 * 保留原 App.tsx 的 UI 渲染逻辑，但移除本地状态管理
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { useGame } from './GameContext';
import { Card, getCardDisplay, isRedSuit } from './types';

// 将牌选项
const TRUMP_OPTIONS = [
  { rank: 14, name: 'A' }, { rank: 13, name: 'K' }, { rank: 12, name: 'Q' },
  { rank: 11, name: 'J' }, { rank: 10, name: '10' }, { rank: 9, name: '9' },
  { rank: 8, name: '8' }, { rank: 7, name: '7' }, { rank: 6, name: '6' },
  { rank: 5, name: '5' }, { rank: 4, name: '4' }, { rank: 3, name: '3' }, { rank: 2, name: '2' },
];

const RANK_NAMES = ['头游', '二游', '三游', '四游', '五游', '末游'];
const AVATARS = ['😎', '😐', '😏', '🤔', '😊', '😆'];

// ─── 渲染工具函数 ─────────────────────────────────────
function rankDisplay(rank: number): string {
  const map: Record<number, string> = {
    2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '10', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'
  };
  return map[rank] || String(rank);
}

function RenderCard({ card, isSelected = false, onClick }: {
  card: Card;
  isSelected?: boolean;
  onClick?: () => void;
}) {
  const isRed = isRedSuit(card.suit);
  const displayText = card.isJoker ? '' : rankDisplay(card.rank as number);
  const suitDisplay = card.isJoker ? '🃏' : card.suit;

  // 大王小王使用图片
  const jokerImage = card.isBigJoker ? '/大王.png' : card.isSmallJoker ? '/小王.png' : null;

  return (
    <div
      className={`card ${isRed ? 'red' : 'black'} ${card.isJoker ? 'joker' : ''} ${isSelected ? 'selected' : ''} ${card.isBigJoker ? 'big-joker' : ''} ${card.isSmallJoker ? 'small-joker' : ''}`}
      onClick={onClick}
      style={{ cursor: onClick ? 'pointer' : undefined }}
    >
      <div className="card-inner">
        {jokerImage ? (
          // 大王小王显示图片
          <img src={jokerImage} alt={card.isBigJoker ? '大王' : '小王'} className="joker-image" />
        ) : (
          <>
            <div className="corner top">
              <span>{displayText}</span>
              <span>{suitDisplay}</span>
            </div>
            <div className="center">
              {suitDisplay}
            </div>
            <div className="corner bottom">
              <span>{displayText}</span>
              <span>{suitDisplay}</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function CardStack({ cards, style, trumpRank }: { cards: Card[]; style?: React.CSSProperties; trumpRank?: number | null }) {
  // 按大小排序显示（大的在前/左）
  const sortedCards = [...cards].sort((a, b) => {
    const getVal = (c: Card) => {
      if (c.isBigJoker) return 100;
      if (c.isSmallJoker) return 99;
      if (trumpRank != null && c.rank === trumpRank) return 98;
      return c.rank;
    };
    return getVal(b) - getVal(a);
  });
  
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', ...style }}>
      {sortedCards.map((card, i) => (
        <div key={card.id} style={{ marginLeft: i > 0 ? '-10px' : '0', zIndex: i }}>
          <RenderCard card={card} />
        </div>
      ))}
    </div>
  );
}

// ─── 主组件 ─────────────────────────────────────────
export default function OnlineGame() {
  const {
    room, myHand, mySortedCards, mySeatIndex, isSpectator,
    selectTrump, playCards, passPlay, startNextRound,
    tributeResist, tributeConfirm, returnCandidatesConfirm, returnConfirm,
    lastError, clearError
  } = useGame();

  const g = room?.game;
  if (!g) return <div className="loading">等待游戏数据...</div>;

  const players = g.players;
  const phase = g.phase;

  // 将我的座位映射到显示 id=0 的位置（保持 UI 兼容性）
  // seatIndex 就是 playerId
  // 本地选牌状态
  const [selectedCardIds, setSelectedCardIds] = useState<string[]>([]);
  const [selectedTributeCardId, setSelectedTributeCardId] = useState<string | null>(null);
  const [selectedReturnCardId, setSelectedReturnCardId] = useState<string | null>(null);
  const [selectedReturnCandidateIds, setSelectedReturnCandidateIds] = useState<string[]>([]);
  const [selectedJokerIds, setSelectedJokerIds] = useState<string[]>([]); // 抗贡选择的王

  // 拖拽理牌相关状态
  const [draggingIndex, setDraggingIndex] = useState<number | null>(null);
  const [draggingIndices, setDraggingIndices] = useState<number[]>([]);
  const dragOverIndex = useRef<number | null>(null);
  
  // 本地理牌顺序 - 使用服务器排序作为初始值，但支持本地拖拽调整
  const [localSortedCards, setLocalSortedCards] = useState<Card[]>([]);
  const prevHandIdsRef = useRef<string[]>([]);
  const prevTrumpRankRef = useRef<number | null | undefined>(undefined);
  const prevHandCountRef = useRef<number>(0); // 记录上一轮手牌数量
  const [hasUserReordered, setHasUserReordered] = useState(false); // 用户是否手动调整过顺序
  
  // 排序函数
  const sortCardsByValue = useCallback((cards: Card[], trumpRank: number | null | undefined) => {
    return [...cards].sort((a, b) => {
      const getVal = (c: Card) => {
        if (c.isBigJoker) return 100;
        if (c.isSmallJoker) return 99;
        if (trumpRank != null && c.rank === trumpRank) return 98;
        return c.rank || 0;
      };
      return getVal(b) - getVal(a);
    });
  }, []);
  
  // 当服务器手牌变化时，更新本地排序
  useEffect(() => {
    const serverCards = mySortedCards.length > 0 ? mySortedCards : myHand;
    const serverCardIds = serverCards.map(c => c.id).join(',');
    const prevCardIds = prevHandIdsRef.current.join(',');
    const trumpChanged = prevTrumpRankRef.current !== g?.trumpRank;
    const handCountChanged = prevHandCountRef.current !== serverCards.length;
    
    // 判断是否是新一轮：
    // 1. 手牌从空开始
    // 2. 手牌ID完全不重叠（新一局发牌）
    // 3. 手牌数量相同但内容完全不同（第二盘发牌）
    const prevIds = prevHandIdsRef.current;
    const newIds = serverCards.map(c => c.id);
    const hasOverlap = prevIds.some(id => newIds.includes(id));
    const isNewRound = prevHandIdsRef.current.length === 0 || !hasOverlap || handCountChanged;
    
    // 手牌ID变化 或 将牌变化（且用户还没手动调整过）时更新
    if (serverCardIds !== prevCardIds || (trumpChanged && !hasUserReordered)) {
      if (isNewRound || trumpChanged) {
        // 首次加载 或 将牌变化 或 新一轮，按正确顺序排序
        setLocalSortedCards(sortCardsByValue(serverCards, g?.trumpRank));
        setHasUserReordered(false); // 重置用户调整标志
      } else {
        // 手牌有变化，保留已有牌的顺序，添加新牌到末尾
        // 保留还存在的牌
        const remaining = localSortedCards.filter(c => newIds.includes(c.id));
        
        // 找出新加入的牌
        const addedCards = serverCards.filter(c => !prevIds.includes(c.id));
        
        // 合并
        if (addedCards.length > 0 || remaining.length !== localSortedCards.length) {
          setLocalSortedCards([...remaining, ...addedCards]);
        }
      }
      prevHandIdsRef.current = serverCards.map(c => c.id);
      prevHandCountRef.current = serverCards.length;
    }
    prevTrumpRankRef.current = g?.trumpRank;
  }, [myHand, mySortedCards, localSortedCards, g?.trumpRank, hasUserReordered, sortCardsByValue]);

  // 当游戏阶段变化时清除选中的牌，避免阶段切换后残留选中状态
  const prevPhaseRef = useRef(phase);
  useEffect(() => {
    if (prevPhaseRef.current !== phase) {
      setSelectedCardIds([]);
      prevPhaseRef.current = phase;
    }
  }, [phase]);

  const isMyTurn = mySeatIndex === g.currentPlayer && phase === 'playing';
  
  // 判断是否"轮到最后出牌者"：当前玩家是最后出牌者且其他活跃玩家都已pass
  const isRoundBackToStarter = (() => {
    if (!isMyTurn || g.firstPlay) return false;
    
    // 找到最后一个有效出牌（非pass）的玩家
    let lastValidPlaySeatIndex: number | null = null;
    for (let i = g.playedCards.length - 1; i >= 0; i--) {
      const rec = g.playedCards[i];
      if (rec.cards && rec.cards.length > 0) {
        lastValidPlaySeatIndex = rec.seatIndex;
        break;
      }
    }
    
    // 如果当前玩家不是最后一个出牌者，不限制
    if (lastValidPlaySeatIndex === null || lastValidPlaySeatIndex !== mySeatIndex) return false;
    
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
    const otherActivePlayers = players.filter(p => !p.isOut && p.seatIndex !== mySeatIndex);
    if (otherActivePlayers.length === 0) return false;
    
    const allOthersPassed = otherActivePlayers.every(p => 
      playsAfterLastValid.some((r: { seatIndex: number; cards: Card[] }) => r.seatIndex === p.seatIndex && r.cards.length === 0)
    );
    
    return allOthersPassed;
  })();

  // ── 拖拽理牌处理 ───────────────────────────────────────
  // 拖拽开始 - 支持多选拖动
  const handleDragStart = useCallback((index: number) => {
    setDraggingIndex(index);
    
    // 如果这张牌被选中，则同时选中所有已选中的牌一起拖动
    const card = localSortedCards[index];
    if (selectedCardIds.includes(card.id)) {
      // 找出所有被选中牌的索引
      const indices = localSortedCards
        .map((c, i) => selectedCardIds.includes(c.id) ? i : -1)
        .filter(i => i !== -1);
      setDraggingIndices(indices);
    } else {
      // 如果拖的是未选中的牌，只拖动这一张
      setDraggingIndices([index]);
    }
  }, [localSortedCards, selectedCardIds]);

  // 拖拽经过
  const handleDragOver = useCallback((e: React.DragEvent, index: number) => {
    e.preventDefault();
    dragOverIndex.current = index;
  }, []);

  // 拖拽结束 - 移动多张牌
  const handleDragEnd = useCallback(() => {
    if (draggingIndex !== null && dragOverIndex.current !== null && draggingIndex !== dragOverIndex.current) {
      const newSorted = [...localSortedCards];
      
      // 获取需要移动的牌（按索引排序）
      const movingIndices = [...draggingIndices].sort((a, b) => a - b);
      const movingCards = movingIndices.map(i => newSorted[i]);
      
      // 计算目标位置
      const targetIndex = dragOverIndex.current;
      
      // 如果目标位置在移动范围之后，需要调整
      let insertIndex = targetIndex;
      const minDragIndex = Math.min(...draggingIndices);
      const maxDragIndex = Math.max(...draggingIndices);
      
      if (targetIndex > maxDragIndex) {
        insertIndex = targetIndex - (maxDragIndex - minDragIndex);
      }
      
      // 先移除所有被拖动的牌（从后往前删除，避免索引变化）
      for (let i = movingIndices.length - 1; i >= 0; i--) {
        newSorted.splice(movingIndices[i], 1);
      }
      
      // 调整插入位置（因为移除后索引会变化）
      let finalInsertIndex = insertIndex;
      if (insertIndex > minDragIndex) {
        // 需要重新计算插入位置
        let offset = 0;
        for (const idx of movingIndices) {
          if (idx < insertIndex) offset++;
        }
        finalInsertIndex = insertIndex - offset;
      }
      
      // 插入到目标位置
      newSorted.splice(finalInsertIndex, 0, ...movingCards);
      setLocalSortedCards(newSorted);
      setHasUserReordered(true); // 标记用户已手动调整顺序
    }
    setDraggingIndex(null);
    setDraggingIndices([]);
    dragOverIndex.current = null;
  }, [draggingIndex, draggingIndices, localSortedCards]);

  const toggleCard = useCallback((card: Card) => {
    // 允许随时选择/取消选择卡牌（用于整理），但出牌按钮只在出牌时可用
    setSelectedCardIds(prev =>
      prev.includes(card.id) ? prev.filter(id => id !== card.id) : [...prev, card.id]
    );
  }, []);

  const handlePlayCards = () => {
    if (selectedCardIds.length === 0) return;
    playCards(selectedCardIds);
    setSelectedCardIds([]);
  };

  const handlePassPlay = () => {
    passPlay();
    setSelectedCardIds([]);
  };

  // ── 进贡辅助 ───────────────────────────────────────
  const ti = g.tributeInfo;
  const isMyTributing = ti && ti.tributors[ti.currentTributorIndex] === mySeatIndex;
  const myJokerCount = myHand.filter(c => c.isJoker).length;
  const canResist = isMyTributing && myJokerCount >= 3;

  // 我的最大牌们（进贡候选）
  const myBestCards: Card[] = (() => {
    if (!isMyTributing || myHand.length === 0) return [];
    const sorted = [...myHand].sort((a, b) => {
      const va = a.isBigJoker ? 100 : a.isSmallJoker ? 99 : a.rank === g.trumpRank ? 98 : (a.rank as number);
      const vb = b.isBigJoker ? 100 : b.isSmallJoker ? 99 : b.rank === g.trumpRank ? 98 : (b.rank as number);
      return vb - va;
    });
    const best = sorted[0];
    if (!best) return [];
    if (best.isJoker) return [best];
    return sorted.filter(c => c.rank === best.rank && !c.isJoker);
  })();

  // ── 还贡辅助 ───────────────────────────────────────
  const validTributes = ti ? ti.tributes.filter(t => !ti.resistedPlayers.includes(t.tributorId)) : [];
  const currentTribute = validTributes[ti?.currentTributorIndex ?? 0];
  const isMyReceiving = currentTribute && currentTribute.receiverId === mySeatIndex;
  const isMyPickReturn = currentTribute && currentTribute.tributorId === mySeatIndex && ti?.returnSubPhase === 'tributor_picking';
  const neededReturnCount = currentTribute
    ? (currentTribute.card.isBigJoker ? 3 : currentTribute.card.isSmallJoker ? 2 : 1)
    : 1;

  const toggleReturnCandidate = (card: Card) => {
    setSelectedReturnCandidateIds(prev => {
      if (prev.includes(card.id)) return prev.filter(id => id !== card.id);
      if (prev.length >= neededReturnCount) return [...prev.slice(1), card.id];
      return [...prev, card.id];
    });
  };

  const handleReturnCandidatesConfirm = () => {
    returnCandidatesConfirm(selectedReturnCandidateIds);
    setSelectedReturnCandidateIds([]);
  };

  const handleReturnConfirm = () => {
    if (!selectedReturnCardId) return;
    returnConfirm(selectedReturnCardId);
    setSelectedReturnCardId(null);
  };

  // ── 视角旋转：计算视觉位置（自己始终在底部=位置0）
  // 视觉位置：0=底部，1=右下，2=右上，3=顶部，4=左上，5=左下
  const getVisualPosition = (actualSeat: number): number => {
    return (actualSeat - mySeatIndex + 6) % 6;
  };

  // ── 渲染玩家座位 ────────────────────────────────────
  const renderPlayerSeat = (seatIndex: number) => {
    const player = players[seatIndex];
    if (!player) return null;

    const isActive = g.currentPlayer === seatIndex && phase === 'playing';
    const showCards = g.playerShowCards[seatIndex] || [];
    const rankIdx = g.finishOrder.indexOf(seatIndex);
    const isMe = seatIndex === mySeatIndex;
    
    // 计算视觉位置（用于确定CSS类名和布局）
    const visualPos = getVisualPosition(seatIndex);

    if (player.isOut && rankIdx >= 0) {
      return (
        <div key={seatIndex} className={`player-seat player-seat-${visualPos} player-finished`}>
          <div className="player-info-wrapper">
            <div className={`player-avatar team${player.team}`}>{AVATARS[seatIndex % 6]}</div>
            <div className={`player-info ${player.disconnected ? 'disconnected' : ''} team${player.team}`}>
              <div className="name">{player.name}{isMe && ' (我)'}{player.disconnected && <span style={{color:'#e74c3c',fontSize:'0.6rem',marginLeft:'4px'}}>断线</span>}</div>
              <div className="rank-badge">{RANK_NAMES[rankIdx]}</div>
              <div className="team-badge">队{player.team}</div>
            </div>
          </div>
        </div>
      );
    }

    const isHorizontal = [1, 2, 4, 5].includes(visualPos);
    const isRight = [1, 2].includes(visualPos);

    const infoBlock = (
      <div className="player-info-wrapper">
        <div className={`player-avatar team${player.team}`}>{AVATARS[seatIndex % 6]}</div>
        <div className={`player-info ${isActive ? 'active' : ''} ${player.disconnected ? 'disconnected' : ''} team${player.team}`}>
          <div className="name">{player.name}{isMe && ' (我)'}{player.disconnected && <span style={{color:'#e74c3c',fontSize:'0.6rem',marginLeft:'4px'}}>断线</span>}</div>
          <div className="cards-left">{player.cardCount}张</div>
          <div className="team-badge">队{player.team}</div>
        </div>
      </div>
    );

    const playedBlock = (
      <div className="player-played">
        {showCards.length > 0
          ? <CardStack cards={showCards} trumpRank={g.trumpRank} />
          : <span style={{ color: '#888', fontSize: '0.6rem' }}>
              {isHorizontal ? '-' : '等待出牌'}
            </span>
        }
      </div>
    );

    if (isHorizontal) {
      return (
        <div key={seatIndex} className={`player-seat player-seat-${visualPos}`}>
          {isRight ? <>{infoBlock}{playedBlock}</> : <>{playedBlock}{infoBlock}</>}
        </div>
      );
    }

    return (
      <div key={seatIndex} className={`player-seat player-seat-${visualPos}`}>
        <div className={`player-avatar team${player.team}`}>{AVATARS[seatIndex % 6]}</div>
        <div className={`player-info ${isActive ? 'active' : ''} ${player.disconnected ? 'disconnected' : ''} team${player.team}`}>
          <div className="name">{player.name}{isMe && ' (我)'}{player.disconnected && <span style={{color:'#e74c3c',fontSize:'0.6rem',marginLeft:'4px'}}>断线</span>}</div>
          <div className="cards-left">{player.cardCount}张</div>
          <div className="team-badge">队{player.team}</div>
        </div>
        {playedBlock}
      </div>
    );
  };

  // ── 渲染手牌 ────────────────────────────────────────
  const handContainerRef = useRef<HTMLDivElement>(null);
  const [cardOverlap, setCardOverlap] = useState(0);
  
  // 显示的牌：直接使用 localSortedCards（它会在初始化时自动排序，也会保留用户的拖拽调整）
  const displayCards = localSortedCards;
  
  // 计算卡牌重叠间距
  useEffect(() => {
    const calculateOverlap = () => {
      if (handContainerRef.current && displayCards.length > 0) {
        const containerWidth = handContainerRef.current.clientWidth;
        // 卡牌宽度：桌面70px，移动端52px
        const isMobile = window.innerWidth <= 600;
        const cardWidth = isMobile ? 52 : 70;
        const cardCount = displayCards.length;
        const gap = 4; // 卡牌之间的最小间距
        const padding = 10; // 容器内边距
        
        // 计算所有牌排一排需要的宽度
        const totalWidth = cardCount * cardWidth + (cardCount - 1) * gap;
        const availableWidth = containerWidth - padding * 2;
        
        if (totalWidth > availableWidth && cardCount > 1) {
          // 需要重叠，计算重叠量使牌均匀分布填满容器
          const overlapAmount = (totalWidth - availableWidth) / (cardCount - 1);
          // 最多重叠到只露出20px
          const maxOverlap = cardWidth - 20;
          setCardOverlap(Math.min(overlapAmount, maxOverlap));
        } else {
          setCardOverlap(0);
        }
      }
    };
    
    calculateOverlap();
    window.addEventListener('resize', calculateOverlap);
    return () => window.removeEventListener('resize', calculateOverlap);
  }, [displayCards.length]);

  const renderMyHand = () => {
    return (
      <div className="sort-cards-container sort-cards-spread" ref={handContainerRef}>
        {displayCards.map((card, index) => {
          const isSelected = selectedCardIds.includes(card.id);
          const isDragging = draggingIndex === index;
          const isMultiDragging = draggingIndices.includes(index);
          const isDragOver = dragOverIndex.current === index;
          
          return (
            <div
              key={card.id}
              className="card-wrapper-stacked"
              draggable
              onDragStart={(e) => {
                e.dataTransfer.effectAllowed = 'move';
                handleDragStart(index);
              }}
              onDragOver={(e) => handleDragOver(e, index)}
              onDragEnd={handleDragEnd}
              onClick={() => toggleCard(card)}
              style={{ 
                marginLeft: index > 0 ? `-${cardOverlap}px` : '0',
                zIndex: isDragging || isMultiDragging ? 100 : (isDragOver ? 50 : (isSelected ? 20 : index)),
                opacity: isMultiDragging && !isDragging ? 0.7 : 1,
                transform: isSelected ? 'translateY(-10px)' : 'none',
              }}
            >
              <RenderCard card={card} isSelected={isSelected} />
            </div>
          );
        })}
      </div>
    );
  };

  // ─── 进贡阶段界面 ──────────────────────────────────
  if (phase === 'tribute' && ti) {
    const currentTributorSeat = ti.tributors[ti.currentTributorIndex];
    const currentTributorPlayer = players[currentTributorSeat];

    return (
      <div className="phase-overlay">
        <h2>进贡阶段</h2>
        {lastError && <div className="lobby-error" onClick={clearError}>⚠ {lastError} <span>✕</span></div>}
        <div className="tribute-info">
          <p className="tribute-status">
            {ti.tributors.map((seat, idx) => {
              const p = players[seat];
              const resisted = ti.resistedPlayers.includes(seat);
              const hasTributed = ti.tributes.some(t => t.tributorId === seat);
              const isCurrent = idx === ti.currentTributorIndex;
              return (
                <span key={seat} className={`tribute-player ${isCurrent ? 'current' : ''} ${resisted ? 'resisted' : ''} ${hasTributed ? 'done' : ''}`}>
                  {p?.name}
                  {resisted && ' (抗贡)'}
                  {hasTributed && !resisted && ' (已进贡)'}
                  {isCurrent && !resisted && !hasTributed && ' (进贡中)'}
                </span>
              );
            })}
          </p>
        </div>

        {isMyTributing && !canResist && (
          <div className="tribute-selection">
            <h3>请选择要进贡的牌（必须进贡最大的牌）</h3>
            {myBestCards.length > 1 && <p className="tribute-hint">你有 {myBestCards.length} 张最大的牌，请选择一张进贡</p>}
            <div className="tribute-countdown">
              剩余时间: <span className={g.tributeCountdown <= 5 ? 'countdown-warning' : ''}>{g.tributeCountdown}秒</span>
            </div>
            <div className="tribute-cards">
              {myBestCards.map(card => (
                <div
                  key={card.id}
                  className={`tribute-card ${selectedTributeCardId === card.id ? 'selected' : ''}`}
                  onClick={() => setSelectedTributeCardId(card.id)}
                >
                  <RenderCard card={card} isSelected={selectedTributeCardId === card.id} />
                </div>
              ))}
            </div>
            <button
              className="btn btn-primary"
              onClick={() => { if (selectedTributeCardId) { tributeConfirm(selectedTributeCardId); setSelectedTributeCardId(null); } }}
              disabled={!selectedTributeCardId}
            >
              确认进贡
            </button>
          </div>
        )}

        {isMyTributing && canResist && (
          <div className="tribute-resist">
            <h3>你有 {myJokerCount} 个王，必须抗贡！</h3>
            <p className="tribute-hint">请选择至少 3 张王展示</p>
            <div className="tribute-countdown">
              剩余时间: <span className={g.tributeCountdown <= 5 ? 'countdown-warning' : ''}>{g.tributeCountdown}秒</span>
            </div>
            <div className="joker-display">
              <div className="joker-cards selectable">
                {myHand.filter(c => c.isJoker).map(card => {
                  const isSel = selectedJokerIds.includes(card.id);
                  return (
                    <div 
                      key={card.id} 
                      className={`joker-card ${isSel ? 'selected' : ''}`}
                      onClick={() => {
                        if (isSel) {
                          setSelectedJokerIds(prev => prev.filter(id => id !== card.id));
                        } else {
                          setSelectedJokerIds(prev => [...prev, card.id]);
                        }
                      }}
                    >
                      <RenderCard card={card} isSelected={isSel} />
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="tribute-buttons">
              <button 
                className="btn btn-primary" 
                onClick={() => { if (selectedJokerIds.length >= 3) tributeResist(selectedJokerIds); }}
                disabled={selectedJokerIds.length < 3}
              >
                确认抗贡（已选 {selectedJokerIds.length} 张）
              </button>
            </div>
          </div>
        )}

        {!isMyTributing && (
          <div className="tribute-waiting">
            <p>等待 {currentTributorPlayer?.name} 进贡...</p>
          </div>
        )}

        {ti.jokerCards.length > 0 && (
          <div className="joker-display">
            <p className="joker-display-title">抗贡展示 {ti.jokerCards.length} 个王！</p>
            <div className="joker-cards">
              {ti.jokerCards.map(card => (
                <div key={card.id} className="joker-card"><RenderCard card={card} /></div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  // ─── 还贡阶段界面 ──────────────────────────────────
  if (phase === 'tribute_return' && ti) {
    if (!currentTribute) return null;
    const receiver = players[currentTribute.receiverId];
    const tributor = players[currentTribute.tributorId];
    const tributeCardDesc = currentTribute.card.isBigJoker ? '大王'
      : currentTribute.card.isSmallJoker ? '小王'
      : getCardDisplay(currentTribute.card);

    if (ti.returnSubPhase === 'selecting_candidates') {
      const ruleText = neededReturnCount === 1
        ? `收到非王牌（${tributeCardDesc}），请直接选 1 张牌还贡`
        : neededReturnCount === 2
        ? `收到小王，请选 2 张候选牌展示给 ${tributor?.name} 挑选`
        : `收到大王，请选 3 张候选牌展示给 ${tributor?.name} 挑选`;

      return (
        <div className="phase-overlay">
          <h2>还贡阶段</h2>
          {lastError && <div className="lobby-error" onClick={clearError}>⚠ {lastError} <span>✕</span></div>}
          <div className="tribute-received">
            <p>{receiver?.name} 收到了来自 {tributor?.name} 的进贡：</p>
            <div className="tribute-card-received">
              <RenderCard card={currentTribute.card} />
              <span className="tribute-card-name">{tributeCardDesc}</span>
            </div>
            <p className="return-rule">{ruleText}</p>
          </div>

          {isMyReceiving && (
            <div className="tribute-selection">
              <div className={`tribute-countdown ${g.returnCountdown <= 5 ? 'countdown-warning' : ''}`}>
                倒计时: {g.returnCountdown}秒
              </div>
              <h3>
                请选择 {neededReturnCount} 张候选牌
                {selectedReturnCandidateIds.length > 0 && (
                  <span style={{ marginLeft: 8, color: '#4caf50' }}>
                    已选 {selectedReturnCandidateIds.length}/{neededReturnCount}
                  </span>
                )}
              </h3>
              <div className="tribute-cards">
                {[...myHand].sort((a, b) => {
                  const getVal = (c: Card) => {
                    if (c.isBigJoker) return 100;
                    if (c.isSmallJoker) return 99;
                    if (g?.trumpRank != null && c.rank === g.trumpRank) return 98;
                    return c.rank || 0;
                  };
                  return getVal(b) - getVal(a);
                }).map(card => {
                  const isSel = selectedReturnCandidateIds.includes(card.id);
                  return (
                    <div
                      key={card.id}
                      className={`tribute-card ${isSel ? 'selected' : ''}`}
                      onClick={() => toggleReturnCandidate(card)}
                    >
                      <RenderCard card={card} isSelected={isSel} />
                    </div>
                  );
                })}
              </div>
              <button
                className="btn btn-primary"
                onClick={handleReturnCandidatesConfirm}
                disabled={selectedReturnCandidateIds.length !== neededReturnCount}
              >
                {neededReturnCount === 1 ? '确认还贡' : `确认展示 ${neededReturnCount} 张候选`}
              </button>
            </div>
          )}

          {!isMyReceiving && (
            <div className="tribute-waiting">
              <p>等待 {receiver?.name} 选择候选牌...</p>
            </div>
          )}
        </div>
      );
    }

    if (ti.returnSubPhase === 'tributor_picking') {
      const candidates = ti.returnCandidates;
      return (
        <div className="phase-overlay">
          <h2>还贡阶段 · 挑选</h2>
          {lastError && <div className="lobby-error" onClick={clearError}>⚠ {lastError} <span>✕</span></div>}
          <div className="tribute-received">
            <p>{receiver?.name} 展示了 {candidates.length} 张候选牌，请 {tributor?.name} 挑选 1 张：</p>
          </div>
          {isMyPickReturn && (
            <div className="tribute-selection">
              <div className={`tribute-countdown ${g.returnCountdown <= 5 ? 'countdown-warning' : ''}`}>
                倒计时: {g.returnCountdown}秒
              </div>
              <h3>请从候选牌中选择 1 张</h3>
              <div className="tribute-cards">
                {candidates.map(card => (
                  <div
                    key={card.id}
                    className={`tribute-card ${selectedReturnCardId === card.id ? 'selected' : ''}`}
                    onClick={() => setSelectedReturnCardId(card.id)}
                  >
                    <RenderCard card={card} isSelected={selectedReturnCardId === card.id} />
                  </div>
                ))}
              </div>
              <button
                className="btn btn-primary"
                onClick={handleReturnConfirm}
                disabled={!selectedReturnCardId}
              >
                确认挑牌
              </button>
            </div>
          )}
          {!isMyPickReturn && (
            <div className="tribute-waiting">
              <p>等待 {tributor?.name} 从候选牌中挑选...</p>
              <div className="tribute-cards" style={{ pointerEvents: 'none', opacity: 0.7 }}>
                {candidates.map(card => (
                  <div key={card.id} className="tribute-card">
                    <RenderCard card={card} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      );
    }

    return null;
  }

  // ─── 回合结束界面 ──────────────────────────────────
  if (phase === 'round_end') {
    const nextRedName = TRUMP_OPTIONS.find(t => t.rank === g.redLevel)?.name || String(g.redLevel);
    const nextBlueName = TRUMP_OPTIONS.find(t => t.rank === g.blueLevel)?.name || String(g.blueLevel);
    const winnerSeat = g.finishOrder[0];
    const firstPlacePlayer = players[winnerSeat];

    return (
      <div className="phase-overlay">
        <h2>本局结束</h2>
        <div className="finish-ranking">
          <h3>排名</h3>
          {g.finishOrder.map((seat, idx) => {
            const p = players[seat];
            if (!p) return null;
            return (
              <div key={seat} className={`rank-item team${p.team}`}>
                <span className="rank-num">{RANK_NAMES[idx]}</span>
                <span className="rank-name">{p.name}</span>
                <span className={`rank-team team${p.team}`}>{p.team === 1 ? '红队' : '蓝队'}</span>
              </div>
            );
          })}
        </div>
        <div className="total-scores">
          <span className="red-score">红队: {g.scores.red}分</span>
          <span className="blue-score">蓝队: {g.scores.blue}分</span>
        </div>
        <div className="next-round-info">
          <p>下一盘首发: {firstPlacePlayer?.name || '-'}</p>
          <p className="next-level-info">
            下盘将牌: <span className="team-red">红打{nextRedName}</span> / <span className="team-blue">蓝打{nextBlueName}</span>
          </p>
        </div>
        {!isSpectator && (
          <button className="btn btn-primary" onClick={startNextRound}>
            下一盘
          </button>
        )}
        {isSpectator && <p style={{ color: '#888' }}>等待玩家开始下一盘...</p>}
      </div>
    );
  }

  // ─── 主游戏界面 ────────────────────────────────────
  const currentPlayerName = players[g.currentPlayer]?.name || '?';
  const trumpName = TRUMP_OPTIONS.find(t => t.rank === g.trumpRank)?.name || '-';

  return (
    <div className="app">
      {/* 错误提示 */}
      {lastError && (
        <div className="lobby-error" onClick={clearError} style={{ position: 'fixed', top: 70, left: '50%', transform: 'translateX(-50%)', zIndex: 9999 }}>
          ⚠ {lastError} <span>✕</span>
        </div>
      )}

      <header className="header">
        <h1>大怪路子</h1>
        <div className="subtitle">
          {phase === 'teaming'
            ? '准备中...'
            : phase === 'dealing'
            ? '发牌中...'
            : `游戏中 | 将牌: ${trumpName}${isSpectator ? ' [观战]' : ''}`}
        </div>
      </header>

      {/* 右上角状态 */}
      {(phase === 'playing' || phase === 'teaming') && g.trumpRank && (
        <div className="game-status-panel">
          <div className="status-row">
            <span className="status-label">将牌:</span>
            <span className="status-value trump">{trumpName}</span>
          </div>
          <div className="status-row">
            <span className="status-label">庄家:</span>
            <span className={`status-value ${g.dealerTeam === 1 ? 'team-red' : 'team-blue'}`}>
              {g.dealerTeam === 1 ? '红方' : '蓝方'}
            </span>
          </div>
          <div className="score-row">
            <span className="red-score">红 {g.scores.red}</span>
            <span className="score-sep">:</span>
            <span className="blue-score">{g.scores.blue} 蓝</span>
          </div>
          <div className="level-row">
            <span className="red-level">红打{TRUMP_OPTIONS.find(t => t.rank === g.redLevel)?.name || g.redLevel}</span>
            <span className="level-sep"> / </span>
            <span className="blue-level">蓝打{TRUMP_OPTIONS.find(t => t.rank === g.blueLevel)?.name || g.blueLevel}</span>
          </div>
        </div>
      )}

      {/* 左上角进贡信息 */}
      {phase === 'playing' && g.lastTributeInfo.length > 0 && (
        <div className="tribute-log-panel">
          <div className="tribute-log-title">本盘进贡</div>
          {g.lastTributeInfo.map((info, idx) => (
            <div key={idx} className="tribute-log-item">
              <span className="tribute-log-from">{info.tributorName}</span>
              <span className="tribute-log-arrow"> → </span>
              <span className="tribute-log-to">{info.receiverName}</span>
              <span className="tribute-log-card"> [{getCardDisplay(info.card)}]</span>
            </div>
          ))}
        </div>
      )}

      {/* 消息栏 */}
      <div className={`message-bar info`}>
        {room?.messages?.slice(-1)[0]?.text || (isMyTurn ? '请出牌' : `等待 ${currentPlayerName} 出牌...`)}
      </div>

      {/* 牌桌 */}
      <div className="game-area">
        <div className="table-surface">
          {[0, 1, 2, 3, 4, 5].map(si => renderPlayerSeat(si))}
          
          {/* 操作按钮区 - 桌子右下角 */}
          {isMyTurn && !isSpectator && (
            <div className="table-action-buttons">
              <div className="countdown-display" style={{ color: g.countdown <= 5 ? '#e74c3c' : '#e8c547' }}>
                {g.countdown}秒
              </div>
              <button
                className="btn btn-primary"
                onClick={handlePlayCards}
                disabled={selectedCardIds.length === 0}
              >
                出牌 ({selectedCardIds.length}张)
              </button>
              <button
                className="btn btn-secondary"
                onClick={handlePassPlay}
                disabled={g.firstPlay || isRoundBackToStarter}
              >
                不出
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => setSelectedCardIds([])}
                disabled={selectedCardIds.length === 0}
              >
                取消
              </button>
            </div>
          )}
          
          {/* 非自己回合的提示 */}
          {!isMyTurn && !isSpectator && (
            <div className="table-wait-info">
              {`等待 ${currentPlayerName} 出牌... (${g.countdown}秒)`}
            </div>
          )}
        </div>
      </div>

      {/* 底部控制区 - 仅手牌 */}
      <div className="bottom-controls">
        <div className="sort-area">
          <div className="sort-area-label">
            {isSpectator
              ? '👁 观战模式 · 无操作权限'
              : isMyTurn
                ? `我的手牌 (${(mySortedCards.length || myHand.length)}张) - 点击选牌，选好后点击"出牌"`
                : `我的手牌 (${(mySortedCards.length || myHand.length)}张) - 可随时整理手牌`
            }
          </div>
          {!isSpectator && renderMyHand()}
          {/* 非出牌回合但有选中卡牌时，显示清空按钮 */}
          {!isMyTurn && selectedCardIds.length > 0 && (
            <button
              className="btn btn-secondary"
              style={{ marginTop: '8px' }}
              onClick={() => setSelectedCardIds([])}
            >
              清空选择 ({selectedCardIds.length}张)
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
