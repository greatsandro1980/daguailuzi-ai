import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Card, Player, GamePhase, GameMessage, CardType, TributeInfo,
  getCardDisplay, isRedSuit
} from './types';
import { 
  createDeck as createDeckFn, 
  shuffleDeck as shuffleDeckFn, 
  dealCards as dealCardsFn,
  sortCards as sortCardsFn,
  recognizeCardType as recognizeCardTypeFn,
  compareCards as compareCardsFn,
  aiSelectCards as aiSelectCardsFn
} from './gameLogic';

// 玩家名称
const PLAYER_NAMES = ['一号', '二号', '三号', '四号', '五号', '六号'];
// 头像
const AVATARS = ['😎', '😐', '😏', '🤔', '😎', '😐'];

// 可选的将牌列表（2-A）
const TRUMP_OPTIONS = [
  { rank: 14, name: 'A' },
  { rank: 13, name: 'K' },
  { rank: 12, name: 'Q' },
  { rank: 11, name: 'J' },
  { rank: 10, name: '10' },
  { rank: 9, name: '9' },
  { rank: 8, name: '8' },
  { rank: 7, name: '7' },
  { rank: 6, name: '6' },
  { rank: 5, name: '5' },
  { rank: 4, name: '4' },
  { rank: 3, name: '3' },
  { rank: 2, name: '2' },
];

function App() {
  const [gamePhase, setGamePhase] = useState<GamePhase>(GamePhase.DEALING);
  const [players, setPlayers] = useState<Player[]>([]);
  const [currentPlayer, setCurrentPlayer] = useState<number>(0);
  const [selectedCards, setSelectedCards] = useState<Card[]>([]);
  const [sortedCards, setSortedCards] = useState<Card[]>([]); // 理牌区
  const [playedCards, setPlayedCards] = useState<{ playerId: number; cards: Card[] }[]>([]);
  const [message, setMessage] = useState<GameMessage>({ id: 0, text: '', type: 'info', timestamp: 0 });
  const [jokers, setJokers] = useState<Card[]>([]);
  const [roundWinner, setRoundWinner] = useState<number | null>(null);
  const [firstPlay, setFirstPlay] = useState<boolean>(true);
  const [playerShowCards, setPlayerShowCards] = useState<Record<number, Card[]>>({}); // 每个玩家面前展示的牌
  const [currentRoundCardCount, setCurrentRoundCardCount] = useState<number | null>(null); // 当前轮次要求出的牌数
  const [roundStarter, setRoundStarter] = useState<number | null>(null); // 当前轮的起始出牌者（上一轮最大牌者）
  const [isProcessing, setIsProcessing] = useState<boolean>(false); // 是否正在处理出牌
  const [countdown, setCountdown] = useState<number>(15); // 倒计时
  const [trumpRank, setTrumpRank] = useState<number | null>(null); // 固定的将牌点数（如打A时=14，打2时=2）
  const [finishOrder, setFinishOrder] = useState<number[]>([]); // 出完牌的排名（ playerId 数组，先出完的排前面）
  const [scores, setScores] = useState<{ red: number; blue: number }>({ red: 0, blue: 0 }); // 总比分
  const [dealerTeam, setDealerTeam] = useState<number>(1); // 庄家队伍（1=红方，2=蓝方）
  const [lastWinnerId, setLastWinnerId] = useState<number | null>(null); // 上一盘头游选手（下一盘首先出牌）
  const [tributeInfo, setTributeInfo] = useState<TributeInfo | null>(null); // 进贡信息
  const [selectedTributeCard, setSelectedTributeCard] = useState<Card | null>(null); // 选中的进贡牌
  const [selectedReturnCard, setSelectedReturnCard] = useState<Card | null>(null); // 选中的还贡牌（进贡方挑选时）
  const [selectedReturnCandidates, setSelectedReturnCandidates] = useState<Card[]>([]); // 受贡方已选中的候选牌列表
  const [pendingTributeInfo, setPendingTributeInfo] = useState<TributeInfo | null>(null); // 待执行的进贡信息（下一盘发牌后执行）
  const [tributeCountdown, setTributeCountdown] = useState<number>(15); // 进贡倒计时
  const [tributeForceNoResist, setTributeForceNoResist] = useState<boolean>(false); // 放弃抗贡，强制选择进贡
  const [returnCountdown, setReturnCountdown] = useState<number>(15); // 还贡倒计时
  const [redLevel, setRedLevel] = useState<number>(2); // 红方当前打几（2-14）
  const [blueLevel, setBlueLevel] = useState<number>(2); // 蓝方当前打几（2-14）
  const [isFirstRound, setIsFirstRound] = useState<boolean>(true); // 是否是第一盘（第一盘需手动选将牌）
  const [lastTributeInfo, setLastTributeInfo] = useState<{tributorName: string; receiverName: string; card: Card}[]>([]); // 本盘进贡信息（显示用）

  // 拖拽相关
  const [draggingIndex, setDraggingIndex] = useState<number | null>(null);
  const [draggingIndices, setDraggingIndices] = useState<number[]>([]); // 记录多选拖拽的索引
  const dragOverIndex = useRef<number | null>(null);
  // 用于打破还贡函数之间的前向声明依赖
  const finishTributePhaseRef = useRef<() => void>(() => {});
  const applyReturnRef = useRef<(t: { tributorId: number; card: Card; receiverId: number }, card: Card) => void>(() => {});
  // 防止 handleGameEnd 在同一盘内被重复调用（React 批量更新 + setTimeout 竞态问题）
  const gameEndCalledRef = useRef<boolean>(false);
  // 防止 AI 还贡 effect 对同一组还贡重复触发（tributeInfo 变化会重新运行 effect）
  // key 格式：`${currentTributorIndex}-${returnSubPhase}`
  const aiReturnProcessedRef = useRef<string>('');

  const showMessage = useCallback((text: string, type: GameMessage['type'] = 'info') => {
    setMessage({ id: Date.now(), text, type, timestamp: Date.now() });
  }, []);

  // 计算得分：头游方得分 = 对方在最后连续几名的数量
  // 例如：红方头游，蓝方在最后连续3名（第4、5、6），红方得3分
  const calculateScore = useCallback((order: number[], playersList: Player[]): { winnerTeam: number | null, score: number } => {
    if (order.length < 6) return { winnerTeam: null, score: 0 };

    // 第一名（头游）的队伍
    const firstPlaceTeam = playersList.find(p => p.id === order[0])?.team;
    if (firstPlaceTeam === undefined) return { winnerTeam: null, score: 0 };

    // 从第六名开始往前数连续的对方数量
    const opponentTeam = firstPlaceTeam === 1 ? 2 : 1;
    let opponentCountFromEnd = 0;
    for (let i = 5; i >= 0; i--) {
      const playerTeam = playersList.find(p => p.id === order[i])?.team;
      if (playerTeam === opponentTeam) {
        opponentCountFromEnd++;
      } else {
        break; // 遇到己方就停止
      }
    }

    return { winnerTeam: firstPlaceTeam, score: opponentCountFromEnd };
  }, []);

  // 获取排名名称
  const getRankName = (rank: number): string => {
    const names = ['头游', '二游', '三游', '四游', '五游', '末游'];
    return names[rank] || `第${rank + 1}名`;
  };

  // 找到下一个未出完的玩家
  const findNextActivePlayer = useCallback((currentId: number, playersList: Player[]): number => {
    let nextId = (currentId + 1) % 6;
    let count = 0;
    while (playersList[nextId]?.isOut && count < 6) {
      nextId = (nextId + 1) % 6;
      count++;
    }
    return nextId;
  }, []);

  // 统计玩家手中的王数量
  const countJokers = useCallback((playerId: number, playersList: Player[]): number => {
    const player = playersList.find(p => p.id === playerId);
    if (!player) return 0;
    return player.hand.filter(c => c.isJoker).length;
  }, []);

  // 获取玩家手中的王（用于抗贡展示）
  const getJokerCards = useCallback((playerId: number, playersList: Player[]): Card[] => {
    const player = playersList.find(p => p.id === playerId);
    if (!player) return [];
    return player.hand.filter(c => c.isJoker);
  }, []);

  // 获取玩家手中最大的牌（用于进贡）
  const getBestCard = useCallback((playerId: number, playersList: Player[]): Card | null => {
    const player = playersList.find(p => p.id === playerId);
    if (!player || player.hand.length === 0) return null;
    
    // 按牌大小排序（大的在前），返回最大的
    const sorted = sortCardsFn(player.hand, jokers);
    return sorted[0]; // 最大的牌在最前面
  }, [jokers]);

  // 获取玩家手中所有最大的牌（同点数不同花色，用于进贡选择）
  // 返回所有与最大牌点数相同的牌
  const getBestCards = useCallback((playerId: number, playersList: Player[]): Card[] => {
    const player = playersList.find(p => p.id === playerId);
    if (!player || player.hand.length === 0) return [];
    
    // 按牌大小排序（大的在前）
    const sorted = sortCardsFn(player.hand, jokers);
    const bestCard = sorted[0];
    
    if (!bestCard) return [];
    
    // 如果是王，只有一张（大王或小王）
    if (bestCard.isJoker) {
      return [bestCard];
    }
    
    // 找所有与最大牌点数相同的牌
    const bestRank = bestCard.rank;
    return sorted.filter(c => c.rank === bestRank && !c.isJoker);
  }, [jokers]);

  // 获取受贡方还贡时需要展示的候选牌数量
  // 规则：进贡的是非王 → 受贡方直接选1张还贡（不需要进贡方挑选）
  //       进贡的是小王 → 受贡方选2张候选，进贡方从中挑1张
  //       进贡的是大王 → 受贡方选3张候选，进贡方从中挑1张
  const getReturnCandidateCount = useCallback((tributeCard: Card): number => {
    if (tributeCard.isBigJoker) return 3;
    if (tributeCard.isSmallJoker) return 2;
    return 1; // 非王：直接还1张，不需要进贡方挑选
  }, []);

  // 获取玩家手中可选的还贡候选牌（返回全部手牌，让玩家从中选）
  const getReturnCardOptions = useCallback((playerId: number, playersList: Player[]): Card[] => {
    const player = playersList.find(p => p.id === playerId);
    if (!player) return [];
    return [...player.hand];
  }, []);

  // 检查是否需要进贡，并初始化进贡信息
  // 进贡规则：
  // - 得分 = 对方末尾连续的人数
  // - 进贡方 = 对方末尾连续的人（按出牌顺序倒序，即末游先贡）
  // - 受贡方 = 己方前面连续的人（按出牌顺序正序，即头游先受）
  const checkAndInitTribute = useCallback((order: number[], playersList: Player[]): TributeInfo | null => {
    if (order.length < 6) return null;

    // 第一名（头游）的队伍
    const firstPlaceTeam = playersList.find(p => p.id === order[0])?.team;
    if (firstPlaceTeam === undefined) return null;

    // 头游方（受贡方）
    const winnerTeam = firstPlaceTeam;
    // 对方（进贡方）
    const loserTeam = winnerTeam === 1 ? 2 : 1;

    // 从第6名（末游）开始往前数连续的对方数量，这就是进贡人数
    const tributors: number[] = [];
    for (let i = 5; i >= 0; i--) {
      const player = playersList.find(p => p.id === order[i]);
      if (player?.team === loserTeam) {
        tributors.push(order[i]); // 按倒序添加，末游在最前面
      } else {
        break; // 遇到己方就停止
      }
    }

    // 如果没有进贡方（得0分），不需要进贡
    if (tributors.length === 0) {
      return null;
    }

    // 受贡方 = 头游方前面连续的人，数量与进贡方相同
    const receivers: number[] = [];
    for (let i = 0; i < tributors.length; i++) {
      const player = playersList.find(p => p.id === order[i]);
      if (player?.team === winnerTeam) {
        receivers.push(order[i]); // 按正序添加，头游在最前面
      } else {
        break; // 理论上不会发生，因为头游方前面肯定都是己方
      }
    }

    // 如果受贡方数量不足（理论上不应该发生），取最小值
    const pairedCount = Math.min(tributors.length, receivers.length);

    return {
      tributors: tributors.slice(0, pairedCount), // 末游在前
      receivers: receivers.slice(0, pairedCount), // 头游在前
      currentTributorIndex: 0,
      currentReceiverIndex: 0,
      tributes: [],
      returns: [],
      canResist: false,
      resistedPlayers: [],
      jokerCards: [], // 用于展示抗贡的王
      returnSubPhase: 'selecting_candidates' as const,
      returnCandidates: []
    };
  }, []);

  // 处理游戏结束，判断是否需要进贡
  const handleGameEnd = useCallback((finalOrder: number[], currentPlayers: Player[]) => {
    // 防止重复调用（ref 守卫优先于 gamePhase，解决 setTimeout 竞态问题）
    if (gameEndCalledRef.current) {
      return;
    }
    if (gamePhase === GamePhase.ROUND_END || gamePhase === GamePhase.TRIBUTE) {
      return;
    }
    gameEndCalledRef.current = true;

    // 计算得分
    const { winnerTeam, score } = calculateScore(finalOrder, currentPlayers);

    setScores(prev => ({
      red: prev.red + (winnerTeam === 1 ? score : 0),
      blue: prev.blue + (winnerTeam === 2 ? score : 0)
    }));

    // 更新庄家
    if (winnerTeam) {
      setDealerTeam(winnerTeam);
    }

    // 根据得分更新获胜方的打几等级（失败方保持不变，得分>0只升一级）
    if (winnerTeam && score > 0) {
      // 最高等级为 A（rank=14），升到 A 后停留在 A
      if (winnerTeam === 1) {
        setRedLevel(prev => Math.min(prev + 1, 14));
      } else {
        setBlueLevel(prev => Math.min(prev + 1, 14));
      }
    }

    // 记录头游选手
    setLastWinnerId(finalOrder[0]);

    // 检查是否需要进贡（下一盘发牌后执行）
    const tributeData = checkAndInitTribute(finalOrder, currentPlayers);

    if (tributeData && tributeData.tributors.length > 0) {
      // 需要进贡，保存进贡信息，下一盘发牌后执行
      setPendingTributeInfo(tributeData);
      
      const tributorNames = tributeData.tributors.map(id => currentPlayers.find(p => p.id === id)?.name).join('、');
      showMessage(`本局结束！${tributorNames} 需要在下一盘进贡`, 'info');
    }

    // 进入回合结束阶段
    setGamePhase(GamePhase.ROUND_END);
    setRoundWinner(finalOrder[0]);

    if (score > 0) {
      showMessage(`${winnerTeam === 1 ? '红队' : '蓝队'}获胜！得${score}分`, 'success');
    } else {
      showMessage(`本局结束！${winnerTeam === 1 ? '红队' : '蓝队'}头游，但不得分`, 'info');
    }
  }, [calculateScore, checkAndInitTribute, showMessage, gamePhase]);

  // 进贡方选择进贡牌（玩家操作）
  const handleTributeSelect = (card: Card) => {
    setSelectedTributeCard(card);
  };

  // 执行抗贡（玩家选择抗贡）
  const confirmResist = useCallback(() => {
    if (!tributeInfo) return;

    const currentTributorId = tributeInfo.tributors[tributeInfo.currentTributorIndex];
    const jokerCount = countJokers(currentTributorId, players);
    const jokerCards = getJokerCards(currentTributorId, players);

    setTributeInfo(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        resistedPlayers: [...prev.resistedPlayers, currentTributorId],
        jokerCards: jokerCards,
        currentTributorIndex: prev.currentTributorIndex + 1,
        currentReceiverIndex: prev.currentReceiverIndex + 1
      };
    });
    showMessage(`你抗贡成功！展示${jokerCount}个王`, 'success');
    setSelectedTributeCard(null);
    setTributeForceNoResist(false);
    setTributeCountdown(15);

    if (tributeInfo.currentTributorIndex + 1 >= tributeInfo.tributors.length) {
      setTimeout(() => proceedToReturnOrEnd(), 1500);
    }
  }, [tributeInfo, players, countJokers, getJokerCards, showMessage]);

  // 确认进贡（玩家选好牌后提交）
  const confirmTribute = useCallback(() => {
    if (!tributeInfo || !selectedTributeCard) return;

    const currentTributorId = tributeInfo.tributors[tributeInfo.currentTributorIndex];
    const currentReceiverId = tributeInfo.receivers[tributeInfo.currentTributorIndex];

    // 正常进贡
    // 从进贡方手牌中移除这张牌
    setPlayers(prev => prev.map(p => {
      if (p.id === currentTributorId) {
        return {
          ...p,
          hand: p.hand.filter(c => c.id !== selectedTributeCard.id),
          cardCount: p.cardCount - 1
        };
      }
      return p;
    }));

    // 如果是玩家进贡，同步更新理牌区
    if (currentTributorId === 0) {
      setSortedCards(prev => prev.filter(c => c.id !== selectedTributeCard.id));
    }

    // 记录进贡
    setTributeInfo(prev => {
      if (!prev) return prev;
      const newTributes = [...prev.tributes, {
        tributorId: currentTributorId,
        card: selectedTributeCard,
        receiverId: currentReceiverId
      }];

      return {
        ...prev,
        tributes: newTributes,
        currentTributorIndex: prev.currentTributorIndex + 1,
        currentReceiverIndex: prev.currentReceiverIndex + 1
      };
    });

    showMessage(`${players.find(p => p.id === currentTributorId)?.name} 进贡 ${getCardDisplay(selectedTributeCard)}`, 'info');
    setSelectedTributeCard(null);
    setTributeForceNoResist(false);
    setTributeCountdown(15);

    // 检查是否还有下一个进贡方
    if (tributeInfo.currentTributorIndex + 1 >= tributeInfo.tributors.length) {
      // 进贡结束，开始还贡
      setTimeout(() => proceedToReturnOrEnd(), 1000);
    }
  }, [tributeInfo, selectedTributeCard, players, showMessage]);

  // 进入还贡阶段或结束
  const proceedToReturnOrEnd = useCallback(() => {
    if (!tributeInfo) return;

    // 检查是否有有效的进贡（非抗贡）
    const validTributes = tributeInfo.tributes.filter(t =>
      !tributeInfo.resistedPlayers.includes(t.tributorId)
    );

    if (validTributes.length > 0) {
      // 有进贡，进入还贡阶段
      aiReturnProcessedRef.current = ''; // 重置 AI 还贡防重复标记
      setTributeInfo(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          currentTributorIndex: 0,
          currentReceiverIndex: 0,
          returnSubPhase: 'selecting_candidates' as const,
          returnCandidates: []
        };
      });
      setReturnCountdown(15); // 重置还贡倒计时
      setGamePhase(GamePhase.TRIBUTE_RETURN);
      showMessage('进贡完成，开始还贡...', 'info');
    } else {
      // 没有有效进贡，直接结束
      finishTributePhaseRef.current();
    }
  }, [tributeInfo, showMessage]);

  // ── 还贡阶段 A：受贡方切换选中/取消候选牌
  const handleReturnCandidateToggle = useCallback((card: Card) => {
    if (!tributeInfo) return;
    const currentTribute = tributeInfo.tributes.filter(
      t => !tributeInfo.resistedPlayers.includes(t.tributorId)
    )[tributeInfo.currentTributorIndex];
    if (!currentTribute) return;

    const needed = getReturnCandidateCount(currentTribute.card);
    setSelectedReturnCandidates(prev => {
      const already = prev.find(c => c.id === card.id);
      if (already) {
        return prev.filter(c => c.id !== card.id); // 取消选中
      }
      if (prev.length >= needed) {
        // 已满额，替换最早选中的
        return [...prev.slice(1), card];
      }
      return [...prev, card];
    });
  }, [tributeInfo, getReturnCandidateCount]);

  // ── 还贡阶段 A：受贡方确认候选牌
  //    非王（needed=1）：直接完成还贡，不需要进贡方挑选
  //    小王/大王（needed=2/3）：进入进贡方挑牌子阶段
  const confirmReturnCandidates = useCallback(() => {
    if (!tributeInfo || selectedReturnCandidates.length === 0) return;

    const validTributes = tributeInfo.tributes.filter(
      t => !tributeInfo.resistedPlayers.includes(t.tributorId)
    );
    const currentTribute = validTributes[tributeInfo.currentTributorIndex];
    if (!currentTribute) return;

    const needed = getReturnCandidateCount(currentTribute.card);

    if (needed === 1) {
      // 非王：直接还贡，不需要进贡方挑牌
      const returnCard = selectedReturnCandidates[0];
      setSelectedReturnCandidates([]);
      setReturnCountdown(15);
      applyReturnRef.current(currentTribute, returnCard);
      // 结束判断依赖 applyReturn 更新后的最新 index（用 setTributeInfo functional update 读取）
      setTributeInfo(prev => {
        if (!prev) return prev;
        const vt = prev.tributes.filter(t => !prev.resistedPlayers.includes(t.tributorId));
        if (prev.currentTributorIndex >= vt.length) {
          setTimeout(() => finishTributePhaseRef.current(), 1000);
        }
        return prev;
      });
    } else {
      // 王：进入进贡方挑牌阶段，将候选牌存入 tributeInfo
      setTributeInfo(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          returnSubPhase: 'tributor_picking' as const,
          returnCandidates: selectedReturnCandidates
        };
      });
      setSelectedReturnCandidates([]);
      setReturnCountdown(15);
      const tributor = players.find(p => p.id === currentTribute.tributorId);
      showMessage(`等待 ${tributor?.name} 从候选牌中挑选...`, 'info');
    }
  }, [tributeInfo, selectedReturnCandidates, getReturnCandidateCount, players, showMessage]);

  // ── 还贡阶段 B：进贡方选择还贡牌（从受贡方给出的候选中挑1张）
  const handleReturnSelect = (card: Card) => {
    setSelectedReturnCard(card);
  };

  // 执行还贡结算（受贡方还出一张牌给进贡方，进贡牌归受贡方）
  const applyReturn = useCallback((
    currentTribute: { tributorId: number; card: Card; receiverId: number },
    returnCard: Card
  ) => {
    const { tributorId, receiverId, card: tributeCard } = currentTribute;

    // 从受贡方手牌中移除还贡牌
    setPlayers(prev => prev.map(p => {
      if (p.id === receiverId) {
        return { ...p, hand: p.hand.filter(c => c.id !== returnCard.id), cardCount: p.cardCount - 1 };
      }
      return p;
    }));
    if (receiverId === 0) {
      setSortedCards(prev => prev.filter(c => c.id !== returnCard.id));
    }

    // 将还贡牌加入进贡方手牌
    setPlayers(prev => prev.map(p => {
      if (p.id === tributorId) {
        return { ...p, hand: [...p.hand, returnCard], cardCount: p.cardCount + 1 };
      }
      return p;
    }));
    if (tributorId === 0) {
      setSortedCards(prev => sortCardsFn([...prev, returnCard], jokers));
    }

    // 将进贡牌加入受贡方手牌
    setPlayers(prev => prev.map(p => {
      if (p.id === receiverId) {
        return { ...p, hand: [...p.hand, tributeCard], cardCount: p.cardCount + 1 };
      }
      return p;
    }));
    if (receiverId === 0) {
      setSortedCards(prev => sortCardsFn([...prev, tributeCard], jokers));
    }

    // 记录还贡，推进到下一组
    setTributeInfo(prev => {
      if (!prev) return prev;
      const newReturns = [...prev.returns, { tributorId, card: returnCard, receiverId }];
      return {
        ...prev,
        returns: newReturns,
        currentTributorIndex: prev.currentTributorIndex + 1,
        returnSubPhase: 'selecting_candidates' as const,
        returnCandidates: []
      };
    });

    showMessage(
      `${players.find(p => p.id === receiverId)?.name} 还贡 ${getCardDisplay(returnCard)} 给 ${players.find(p => p.id === tributorId)?.name}`,
      'info'
    );
    setSelectedReturnCard(null);

    // 注意：还贡是否结束由调用方（confirmReturnCandidates / confirmReturn / AI useEffect）负责判断
  }, [players, jokers, showMessage]);
  // 同步到 ref
  applyReturnRef.current = applyReturn;

  // ── 还贡阶段 B：进贡方确认挑牌
  const confirmReturn = useCallback(() => {
    if (!tributeInfo || !selectedReturnCard) return;
    setReturnCountdown(15);

    const validTributes = tributeInfo.tributes.filter(
      t => !tributeInfo.resistedPlayers.includes(t.tributorId)
    );
    const currentTribute = validTributes[tributeInfo.currentTributorIndex];
    if (!currentTribute) return;

    applyReturn(currentTribute, selectedReturnCard);
    // 结束判断依赖 applyReturn 更新后的最新 index
    setTributeInfo(prev => {
      if (!prev) return prev;
      const vt = prev.tributes.filter(t => !prev.resistedPlayers.includes(t.tributorId));
      if (prev.currentTributorIndex >= vt.length) {
        setTimeout(() => finishTributePhaseRef.current(), 1000);
      }
      return prev;
    });
  }, [tributeInfo, selectedReturnCard, applyReturn]);

  // 结束进贡/还贡阶段
  const finishTributePhase = useCallback(() => {
    // 保存本盘进贡信息（用于 ROUND_END 界面显示）
    if (tributeInfo && tributeInfo.tributes.length > 0) {
      const validTributes = tributeInfo.tributes.filter(
        t => !tributeInfo.resistedPlayers.includes(t.tributorId)
      );
      const formattedInfo = validTributes.map(t => ({
        tributorName: players.find(p => p.id === t.tributorId)?.name || '',
        receiverName: players.find(p => p.id === t.receiverId)?.name || '',
        card: t.card
      }));
      setLastTributeInfo(formattedInfo);
    } else {
      setLastTributeInfo([]);
    }

    setTributeInfo(null);

    // 进贡结束后自动设将牌（庄家方等级）
    const autoRank = dealerTeam === 1 ? redLevel : blueLevel;
    const trumpName = TRUMP_OPTIONS.find(t => t.rank === autoRank)?.name || String(autoRank);
    setTrumpRank(autoRank);

    // 对当前手牌按将牌重新排序
    setSortedCards(prev => sortCardsFn(prev, jokers, autoRank));
    setPlayers(prev => prev.map(p => ({
      ...p,
      hand: sortCardsFn(p.hand, jokers, autoRank)
    })));

    showMessage(`进贡完成！本局${dealerTeam === 1 ? '红方' : '蓝方'}打${trumpName}，自动设将牌`, 'info');

    setTimeout(() => {
      setGamePhase(GamePhase.PLAYING);
      // 首发已在 startNextRound 中设置好
    }, 1000);
  }, [showMessage, dealerTeam, redLevel, blueLevel, jokers, tributeInfo, players]);
  // 同步到 ref，供前向引用的函数使用
  finishTributePhaseRef.current = finishTributePhase;

  // AI自动进贡
  useEffect(() => {
    if (gamePhase !== GamePhase.TRIBUTE || !tributeInfo) return;

    const currentTributorId = tributeInfo.tributors[tributeInfo.currentTributorIndex];
    if (currentTributorId === undefined) {
      // 进贡结束，检查是否需要还贡
      proceedToReturnOrEnd();
      return;
    }

    // 如果是玩家（0号），等待玩家操作
    if (currentTributorId === 0) return;

    // 检查是否可以抗贡
    const jokerCount = countJokers(currentTributorId, players);
    if (jokerCount >= 3) {
      // 抗贡，获取王的牌用于展示
      const jokerCards = getJokerCards(currentTributorId, players);
      setTimeout(() => {
        setTributeInfo(prev => {
          if (!prev) return prev;
          return {
            ...prev,
            resistedPlayers: [...prev.resistedPlayers, currentTributorId],
            jokerCards: jokerCards,
            currentTributorIndex: prev.currentTributorIndex + 1,
            currentReceiverIndex: prev.currentReceiverIndex + 1
          };
        });
        showMessage(`${players.find(p => p.id === currentTributorId)?.name} 抗贡成功！展示${jokerCount}个王`, 'success');

        // 检查是否还有下一个
        if (tributeInfo.currentTributorIndex + 1 >= tributeInfo.tributors.length) {
          setTimeout(() => proceedToReturnOrEnd(), 1000);
        }
      }, 1000);
      return;
    }

    // AI自动选择最大的牌进贡
    const bestCard = getBestCard(currentTributorId, players);
    if (bestCard) {
      setTimeout(() => {
        // 从AI手牌中移除
        setPlayers(prev => prev.map(p => {
          if (p.id === currentTributorId) {
            return {
              ...p,
              hand: p.hand.filter(c => c.id !== bestCard.id),
              cardCount: p.cardCount - 1
            };
          }
          return p;
        }));

        const currentReceiverId = tributeInfo.receivers[tributeInfo.currentTributorIndex];

        // 记录进贡
        setTributeInfo(prev => {
          if (!prev) return prev;
          const newTributes = [...prev.tributes, {
            tributorId: currentTributorId,
            card: bestCard,
            receiverId: currentReceiverId
          }];

          return {
            ...prev,
            tributes: newTributes,
            currentTributorIndex: prev.currentTributorIndex + 1,
            currentReceiverIndex: prev.currentReceiverIndex + 1
          };
        });

        showMessage(`${players.find(p => p.id === currentTributorId)?.name} 进贡 ${getCardDisplay(bestCard)}`, 'info');

        // 检查是否还有下一个
        if (tributeInfo.currentTributorIndex + 1 >= tributeInfo.tributors.length) {
          setTimeout(() => proceedToReturnOrEnd(), 1000);
        }
      }, 1000);
    }
  }, [gamePhase, tributeInfo, players, countJokers, getBestCard, showMessage, proceedToReturnOrEnd]);

  // 玩家进贡倒计时 - 15秒自动随机选择最大的牌进贡
  useEffect(() => {
    if (gamePhase !== GamePhase.TRIBUTE || !tributeInfo) {
      setTributeCountdown(15);
      return;
    }

    const currentTributorId = tributeInfo.tributors[tributeInfo.currentTributorIndex];
    if (currentTributorId !== 0) {
      setTributeCountdown(15);
      return; // 不是玩家的回合
    }

    // 检查是否可以抗贡且未选择放弃抗贡
    const canResist = countJokers(0, players) >= 3;
    if (canResist && !tributeForceNoResist) {
      return; // 可以抗贡时，不启动倒计时，让玩家选择
    }

    const timer = setInterval(() => {
      setTributeCountdown(prev => {
        if (prev <= 1) {
          // 时间到，自动随机选择最大的牌进贡
          const bestCards = getBestCards(0, players);
          if (bestCards.length > 0) {
            // 随机选择一张最大的牌
            const randomCard = bestCards[Math.floor(Math.random() * bestCards.length)];
            setSelectedTributeCard(randomCard);
            setTimeout(() => {
              // 自动确认进贡
              setTributeInfo(prev => {
                if (!prev) return prev;
                const currentTributorId = prev.tributors[prev.currentTributorIndex];
                const currentReceiverId = prev.receivers[prev.currentReceiverIndex];

                // 从手牌移除
                setPlayers(pList => pList.map(p => {
                  if (p.id === currentTributorId) {
                    return {
                      ...p,
                      hand: p.hand.filter(c => c.id !== randomCard.id),
                      cardCount: p.cardCount - 1
                    };
                  }
                  return p;
                }));

                // 更新理牌区
                setSortedCards(cards => cards.filter(c => c.id !== randomCard.id));

                // 记录进贡
                const newTributes = [...prev.tributes, {
                  tributorId: currentTributorId,
                  card: randomCard,
                  receiverId: currentReceiverId
                }];

                // 检查是否还有下一个
                if (prev.currentTributorIndex + 1 >= prev.tributors.length) {
                  setTimeout(() => proceedToReturnOrEnd(), 1000);
                }

                return {
                  ...prev,
                  tributes: newTributes,
                  currentTributorIndex: prev.currentTributorIndex + 1,
                  currentReceiverIndex: prev.currentReceiverIndex + 1
                };
              });
              showMessage(`超时自动进贡 ${getCardDisplay(randomCard)}`, 'warning');
              setTributeForceNoResist(false);
            }, 100);
          }
          return 15;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [gamePhase, tributeInfo, players, tributeForceNoResist, countJokers, getBestCards, showMessage, proceedToReturnOrEnd]);

  // 玩家还贡倒计时 - 15秒自动选择最小的牌还贡
  useEffect(() => {
    if (gamePhase !== GamePhase.TRIBUTE_RETURN || !tributeInfo) {
      setReturnCountdown(15);
      return;
    }

    const validTributes = tributeInfo.tributes.filter(t =>
      !tributeInfo.resistedPlayers.includes(t.tributorId)
    );
    if (validTributes.length === 0) {
      setReturnCountdown(15);
      return;
    }

    const currentTribute = validTributes[tributeInfo.currentTributorIndex];
    if (!currentTribute) {
      setReturnCountdown(15);
      return;
    }

    const needed = getReturnCandidateCount(currentTribute.card);

    // 阶段A：受贡方是玩家才启动倒计时
    if (tributeInfo.returnSubPhase === 'selecting_candidates') {
      if (currentTribute.receiverId !== 0) {
        setReturnCountdown(15);
        return;
      }
      const timer = setInterval(() => {
        setReturnCountdown(prev => {
          if (prev <= 1) {
            // 超时：自动选最小的 needed 张牌作为候选，然后直接确认提交
            const player = players.find(p => p.id === 0);
            if (player && player.hand.length > 0) {
              const sorted = sortCardsFn(player.hand, jokers, trumpRank || undefined);
              const autoCandidates = sorted.slice(sorted.length - needed);
              showMessage(`超时自动选择候选牌`, 'warning');
              // 直接调用确认逻辑，不经过 selectedReturnCandidates state
              setTimeout(() => {
                if (needed === 1) {
                  applyReturnRef.current(currentTribute, autoCandidates[0]);
                  setTributeInfo(prev => {
                    if (!prev) return prev;
                    const vt = prev.tributes.filter(t => !prev.resistedPlayers.includes(t.tributorId));
                    if (prev.currentTributorIndex + 1 >= vt.length) {
                      setTimeout(() => finishTributePhaseRef.current(), 1000);
                    }
                    return prev; // applyReturn 内部已更新 index
                  });
                } else {
                  setTributeInfo(prev => {
                    if (!prev) return prev;
                    return {
                      ...prev,
                      returnSubPhase: 'tributor_picking' as const,
                      returnCandidates: autoCandidates
                    };
                  });
                  setReturnCountdown(15);
                }
              }, 100);
            }
            return 15;
          }
          return prev - 1;
        });
      }, 1000);
      return () => clearInterval(timer);
    }

    // 阶段B：进贡方是玩家才启动倒计时
    if (tributeInfo.returnSubPhase === 'tributor_picking') {
      if (currentTribute.tributorId !== 0) {
        setReturnCountdown(15);
        return;
      }
      const candidates = tributeInfo.returnCandidates;
      const timer = setInterval(() => {
        setReturnCountdown(prev => {
          if (prev <= 1) {
            // 超时：自动选候选牌里第一张，然后直接确认提交
            if (candidates.length > 0) {
              const autoCard = candidates[0];
              showMessage(`超时自动挑选 ${getCardDisplay(autoCard)}`, 'warning');
              setTimeout(() => {
                applyReturnRef.current(currentTribute, autoCard);
                setTributeInfo(prev => {
                  if (!prev) return prev;
                  const vt = prev.tributes.filter(t => !prev.resistedPlayers.includes(t.tributorId));
                  if (prev.currentTributorIndex >= vt.length) {
                    setTimeout(() => finishTributePhaseRef.current(), 1000);
                  }
                  return prev;
                });
              }, 100);
            }
            return 15;
          }
          return prev - 1;
        });
      }, 1000);
      return () => clearInterval(timer);
    }

    return;
  }, [gamePhase, tributeInfo, players, jokers, trumpRank, showMessage, getReturnCandidateCount]);

  // AI自动还贡
  useEffect(() => {
    if (gamePhase !== GamePhase.TRIBUTE_RETURN || !tributeInfo) return;

    const validTributes = tributeInfo.tributes.filter(t =>
      !tributeInfo.resistedPlayers.includes(t.tributorId)
    );

    if (tributeInfo.currentTributorIndex >= validTributes.length) {
      finishTributePhaseRef.current();
      return;
    }

    const currentTribute = validTributes[tributeInfo.currentTributorIndex];
    if (!currentTribute) {
      finishTributePhaseRef.current();
      return;
    }

    const { tributorId, receiverId } = currentTribute;
    const needed = getReturnCandidateCount(currentTribute.card);

    // ── 防重复触发：同一组（index + subPhase）只处理一次 ──────────────
    const processKey = `${tributeInfo.currentTributorIndex}-${tributeInfo.returnSubPhase}`;
    if (aiReturnProcessedRef.current === processKey) return;

    // ── 阶段A：受贡方选候选牌 ──────────────────────
    if (tributeInfo.returnSubPhase === 'selecting_candidates') {
      // 受贡方是玩家，等待玩家操作（不消耗 key，保持等待）
      if (receiverId === 0) return;

      // 标记本组已开始处理
      aiReturnProcessedRef.current = processKey;

      // AI 受贡方自动选候选牌
      const allCards = getReturnCardOptions(receiverId, players);
      const sorted = sortCardsFn(allCards, jokers, undefined);
      const candidates = sorted.slice(sorted.length - needed); // 取最小的 needed 张

      if (candidates.length > 0) {
        setTimeout(() => {
          if (needed === 1) {
            // 非王：AI直接还贡，不需要进贡方挑选
            applyReturn(currentTribute, candidates[0]);
            showMessage(`${players.find(p => p.id === receiverId)?.name} 还贡给 ${players.find(p => p.id === tributorId)?.name}`, 'info');
            // 结束判断读最新 index
            setTributeInfo(prev => {
              if (!prev) return prev;
              const vt = prev.tributes.filter(t => !prev.resistedPlayers.includes(t.tributorId));
              if (prev.currentTributorIndex >= vt.length) {
                setTimeout(() => finishTributePhaseRef.current(), 1000);
              }
              return prev;
            });
          } else {
            // 王：AI将候选牌公开，等进贡方（可能是玩家/AI）挑选
            setTributeInfo(prev => {
              if (!prev) return prev;
              return {
                ...prev,
                returnSubPhase: 'tributor_picking' as const,
                returnCandidates: candidates
              };
            });
            const receiver = players.find(p => p.id === receiverId);
            showMessage(`${receiver?.name} 展示了 ${needed} 张候选牌，等待进贡方挑选...`, 'info');
          }
        }, 1000);
      }
    }

    // ── 阶段B：进贡方挑牌 ──────────────────────────
    else if (tributeInfo.returnSubPhase === 'tributor_picking') {
      // 进贡方是玩家，等待玩家操作（不消耗 key，保持等待）
      if (tributorId === 0) return;

      // 标记本组已开始处理
      aiReturnProcessedRef.current = processKey;

      // AI 进贡方从候选牌中挑最大的
      const candidates = tributeInfo.returnCandidates;
      if (candidates.length === 0) return;

      const sortedCandidates = sortCardsFn(candidates, jokers, undefined);
      const bestCandidate = sortedCandidates[0]; // 取最大的

      setTimeout(() => {
        applyReturn(currentTribute, bestCandidate);
        showMessage(
          `${players.find(p => p.id === tributorId)?.name} 从候选中挑选了 ${getCardDisplay(bestCandidate)}`,
          'info'
        );
        // 结束判断读最新 index
        setTributeInfo(prev => {
          if (!prev) return prev;
          const vt = prev.tributes.filter(t => !prev.resistedPlayers.includes(t.tributorId));
          if (prev.currentTributorIndex >= vt.length) {
            setTimeout(() => finishTributePhaseRef.current(), 1000);
          }
          return prev;
        });
      }, 1000);
    }
  }, [gamePhase, tributeInfo, players, getReturnCardOptions, getReturnCandidateCount,
      applyReturn, showMessage, jokers]);

  // 初始化游戏（用于完全重置）
  const initGame = useCallback(() => {
    // 重置防重入守卫
    gameEndCalledRef.current = false;
    aiReturnProcessedRef.current = '';

    const deck = createDeckFn();
    const shuffled = shuffleDeckFn(deck);
    const hands = dealCardsFn(shuffled, 6);

    const jokerCards = shuffled.filter(c => c.isJoker);
    setJokers(jokerCards);

    // 玩家手牌按大小排序
    const sortedHand = sortCardsFn(hands[0], jokerCards);

    const newPlayers: Player[] = Array.from({ length: 6 }, (_, i) => ({
      id: i,
      name: PLAYER_NAMES[i],
      hand: i === 0 ? sortedHand : sortCardsFn(hands[i], jokerCards),
      position: i,
      team: i % 2 === 0 ? 1 : 2,
      isAI: true,
      isOut: false,
      cardCount: hands[i].length,
      hasAsked: false
    }));

    // 初始化理牌区 - 复制玩家手牌
    setSortedCards([...sortedHand]);

    setPlayers(newPlayers);
    setSelectedCards([]);
    setGamePhase(GamePhase.TEAMING);
    setPlayedCards([]);
    setPlayerShowCards({}); // 初始化玩家展示牌
    setRoundWinner(null);
    setFirstPlay(true);
    setCurrentRoundCardCount(null); // 初始化牌数要求
    setRoundStarter(0); // 初始化出牌者
    setTrumpRank(null); // 重置将牌，等待玩家选择
    setFinishOrder([]); // 重置排名
    setDealerTeam(1); // 一号（红方）首发，为庄家
    setLastWinnerId(null);
    setRedLevel(2);   // 重置红方打几
    setBlueLevel(2);  // 重置蓝方打几
    setIsFirstRound(true); // 标记第一盘

    showMessage('请选择本局将牌...', 'info');
  }, [showMessage]);

  // 选择将牌（打A、打2等）
  const selectTrump = (rank: number) => {
    setTrumpRank(rank);
    setIsFirstRound(false); // 第一盘将牌选定后，后续不再手选
    const trumpName = TRUMP_OPTIONS.find(t => t.rank === rank)?.name || String(rank);
    showMessage(`本局打${trumpName}，${trumpName}是将牌！`, 'success');

    // 重新排序理牌区（考虑将牌）
    setSortedCards(prev => sortCardsFn(prev, jokers, rank));

    // 重新排序所有玩家手牌
    setPlayers(prev => prev.map(p => ({
      ...p,
      hand: sortCardsFn(p.hand, jokers, rank)
    })));

    // 延迟后开始游戏
    setTimeout(() => {
      const team1 = players.filter(p => p.team === 1).map(p => p.name).join('、');
      const team2 = players.filter(p => p.team === 2).map(p => p.name).join('、');
      showMessage(`红队：${team1} | 蓝队：${team2}`, 'info');

      setTimeout(() => {
        setGamePhase(GamePhase.PLAYING);
        // 使用 lastWinnerId 或默认 0 作为首发
        const starterId = lastWinnerId ?? 0;
        setCurrentPlayer(starterId);
        setRoundStarter(starterId);
        const starterName = players.find(p => p.id === starterId)?.name || '一号';
        showMessage(`${starterName}先出牌`, 'info');
      }, 1500);
    }, 1000);
  };

  // 选择/取消选择卡牌（从理牌区选）
  const toggleCardSelection = (card: Card) => {
    if (gamePhase !== GamePhase.PLAYING || currentPlayer !== 0) return;

    setSelectedCards(prev => {
      const isSelected = prev.some(c => c.id === card.id);
      if (isSelected) {
        return prev.filter(c => c.id !== card.id);
      } else {
        return [...prev, card];
      }
    });
  };

  // 拖拽开始 - 支持多选拖动（随时可以理牌）
  const handleDragStart = (index: number) => {
    if (gamePhase !== GamePhase.PLAYING) return;
    setDraggingIndex(index);
    
    // 如果这张牌被选中，则同时选中所有已选中的牌一起拖动
    const card = sortedCards[index];
    if (selectedCards.some(c => c.id === card.id)) {
      // 找出所有被选中牌的索引
      const indices = sortedCards
        .map((c, i) => selectedCards.some(s => s.id === c.id) ? i : -1)
        .filter(i => i !== -1);
      setDraggingIndices(indices);
    } else {
      // 如果拖的是未选中的牌，只拖动这一张
      setDraggingIndices([index]);
    }
  };

  // 拖拽经过
  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    dragOverIndex.current = index;
  };

  // 拖拽结束 - 移动多张牌
  const handleDragEnd = () => {
    if (draggingIndex !== null && dragOverIndex.current !== null && draggingIndex !== dragOverIndex.current) {
      const newSorted = [...sortedCards];
      
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
      
      // 先移除所有被拖动的牌
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
      setSortedCards(newSorted);
    }
    setDraggingIndex(null);
    setDraggingIndices([]);
    dragOverIndex.current = null;
  };

  // 出牌
  const playCards = useCallback(() => {
    // 如果正在处理出牌，跳过
    if (isProcessing) return;

    setIsProcessing(true);
    const cardsToPlay = [...selectedCards];

    if (cardsToPlay.length === 0) {
      showMessage('请选择要出的牌', 'warning');
      setIsProcessing(false);
      return;
    }

    const cardType = recognizeCardTypeFn(cardsToPlay, jokers, trumpRank);
    if (!cardType) {
      showMessage('请选择有效的牌型', 'warning');
      setIsProcessing(false);
      return;
    }

    // 检查出牌数量是否匹配当前轮次
    if (currentRoundCardCount !== null && cardsToPlay.length !== currentRoundCardCount) {
      showMessage(`本轮需要出${currentRoundCardCount}张牌`, 'warning');
      setIsProcessing(false);
      return;
    }

    if (firstPlay) {
      // 首轮可以出：单张、对子、三张、5张（顺子、同花、三带二、四带一、五同、同花顺）
      const firstPlayValidTypes = [
        CardType.SINGLE,
        CardType.PAIR,
        CardType.TRIPLE,
        CardType.STRAIGHT,      // 杂顺 5张
        CardType.FLUSH,         // 同花 5张
        CardType.THREE_WITH_TWO, // 三带二 5张
        CardType.FOUR_WITH_ONE,  // 四带一 5张
        CardType.FIVE_OF_KIND,   // 五同 5张
        CardType.STRAIGHT_FLUSH  // 同花顺 5张
      ];
      if (!firstPlayValidTypes.includes(cardType)) {
        showMessage('首轮只能出单张、对子、三张或五张', 'warning');
        setIsProcessing(false);
        return;
      }
      setFirstPlay(false);
      setCurrentRoundCardCount(cardsToPlay.length); // 记录首轮出牌数量
    }

    if (playedCards.length > 0) {
      // 找最后一个实际出牌的记录（不是"不出"）
      const lastActualPlay = [...playedCards].reverse().find(p => p.cards.length > 0);
      if (lastActualPlay) {
        const lastType = recognizeCardTypeFn(lastActualPlay.cards, jokers, trumpRank);

        // compareCards返回: 1=能压过, 0=一样大, -1=不能压过
        // 只有返回 1（更大）才能出牌，相等或更小都不能
        const compareResult = compareCardsFn(cardsToPlay, lastActualPlay.cards, jokers, trumpRank);
        if (lastType && compareResult <= 0) {
          showMessage('牌型或点数不够大，无法压过', 'warning');
          setIsProcessing(false);
          return;
        }
      }
    }

    // 从理牌区移除
    setSortedCards(prev => prev.filter(c => !cardsToPlay.some(s => s.id === c.id)));

    // 更新玩家牌张数
    setPlayers(prev => prev.map(p => {
      if (p.id === 0) {
        return {
          ...p,
          hand: p.hand.filter(c => !cardsToPlay.some(s => s.id === c.id)),
          cardCount: p.hand.filter(c => !cardsToPlay.some(s => s.id === c.id)).length
        };
      }
      return p;
    }));

    setPlayedCards(prev => [...prev, { playerId: 0, cards: cardsToPlay }]);
    // 更新玩家面前展示的牌
    setPlayerShowCards(prev => ({ ...prev, 0: cardsToPlay }));

    showMessage(`你出了 ${cardsToPlay.map(c => getCardDisplay(c)).join(' ')}`, 'success');

    // 检查是否出完牌
    const remainingHand = sortedCards.filter(c => !cardsToPlay.some(s => s.id === c.id));
    const isPlayerOut = remainingHand.length === 0;

    if (isPlayerOut) {
      // 玩家出完牌，记录排名
      const newRank = finishOrder.length;
      setFinishOrder(prev => [...prev, 0]);
      // 标记为出局
      setPlayers(prev => prev.map(p => p.id === 0 ? { ...p, isOut: true } : p));
      showMessage(`你出完牌了！获得 ${getRankName(newRank)}`, 'success');

      // 检查是否所有人都出完牌
      const newFinishOrder = [...finishOrder, 0];
      if (newFinishOrder.length === 6) {
        // 本局结束，检查是否需要进贡
        const updatedPlayersList = players.map(p => p.id === 0 ? { ...p, isOut: true } : p);
        handleGameEnd(newFinishOrder, updatedPlayersList);
      }
    }

    setSelectedCards([]);

    // 找到下一个未出完的玩家
    const updatedPlayers = players.map(p => p.id === 0 ? { ...p, isOut: isPlayerOut } : p);
    const nextPlayer = findNextActivePlayer(0, updatedPlayers);
    setCurrentPlayer(nextPlayer);
    setIsProcessing(false);
  }, [selectedCards, sortedCards, players, playedCards, jokers, firstPlay, showMessage, roundStarter, finishOrder.length, findNextActivePlayer]);

  // 检查一轮是否结束的辅助函数
  // 返回：{ isRoundEnd: boolean, nextStarter: number | null }
  // 如果一轮结束，nextStarter 是下一轮首发玩家
  const checkRoundEnd = useCallback((currentPlayerId: number, playersList: Player[]): { isRoundEnd: boolean; nextStarter: number | null } => {
    // 获取最后一个实际出牌的玩家
    const lastActualPlay = playedCards.length > 0
      ? [...playedCards].reverse().find(p => p.cards.length > 0)
      : null;

    if (!lastActualPlay) return { isRoundEnd: false, nextStarter: null };

    const lastPlayerId = lastActualPlay.playerId;
    const lastPlayer = playersList.find(p => p.id === lastPlayerId);

    // 如果当前玩家就是最后一个出牌者，说明其他人都不出，一轮结束
    if (currentPlayerId === lastActualPlay.playerId) {
      // 如果最后一个出牌者已经出完牌，由其下家接风
      if (lastPlayer?.isOut) {
        const nextActivePlayer = findNextActivePlayer(lastPlayerId, playersList);
        return { isRoundEnd: true, nextStarter: nextActivePlayer };
      }

      // 否则由最大牌者继续出牌
      return { isRoundEnd: true, nextStarter: lastPlayerId };
    }

    // 如果最后一个出牌者已经出完牌，一轮也应该结束（其他人都不出，或者轮回到已出完的人）
    // 此时由下家接风
    if (lastPlayer?.isOut) {
      const nextActivePlayer = findNextActivePlayer(lastPlayerId, playersList);
      return { isRoundEnd: true, nextStarter: nextActivePlayer };
    }

    return { isRoundEnd: false, nextStarter: null };
  }, [playedCards, findNextActivePlayer]);

  // 不出
  const passPlay = useCallback(() => {
    if (currentPlayer !== 0 || isProcessing) return;

    setIsProcessing(true);
    setPlayedCards(prev => [...prev, { playerId: 0, cards: [] }]);
    // 不出时保持之前的牌在面前显示，不清空
    showMessage('你选择不出', 'info');

    // 检查是否一轮结束
    const roundEndResult = checkRoundEnd(0, players);
    if (roundEndResult.isRoundEnd && roundEndResult.nextStarter !== null) {
      // 一轮结束，开始新一轮
      const nextStarterId = roundEndResult.nextStarter;
      const lastActualPlay = [...playedCards].reverse().find(p => p.cards.length > 0);
      const lastPlayerId = lastActualPlay?.playerId ?? 0;
      const lastPlayer = players.find(p => p.id === lastPlayerId);

      // 显示消息
      if (lastPlayer?.isOut && lastPlayerId !== nextStarterId) {
        showMessage(`一轮结束！${lastPlayer.name} 出完牌，${players[nextStarterId]?.name} 接风出牌`, 'info');
      } else {
        showMessage(`一轮结束！${players[nextStarterId]?.name} 最大，下一轮从ta开始`, 'info');
      }

      // 下一轮首发
      setRoundStarter(nextStarterId);
      setPlayedCards([]);
      setPlayerShowCards({});
      setCurrentRoundCardCount(null);
      setFirstPlay(true);
      setCurrentPlayer(nextStarterId);
      setIsProcessing(false);
      return;
    }

    // 继续询问下一个玩家（跳过已出完的玩家）
    const nextPlayer = findNextActivePlayer(0, players);
    setCurrentPlayer(nextPlayer);
    setIsProcessing(false);
  }, [currentPlayer, playedCards, showMessage, checkRoundEnd, players, findNextActivePlayer]);

  // 玩家倒计时 - 15秒自动出牌/不出
  useEffect(() => {
    if (gamePhase !== GamePhase.PLAYING || currentPlayer !== 0 || isProcessing) {
      setCountdown(15); // 重置倒计时
      return;
    }

    // 检查是否一轮结束（牌权回到上一个出牌者）
    const roundEndResult = checkRoundEnd(0, players);
    if (roundEndResult.isRoundEnd && roundEndResult.nextStarter !== null) {
      // 一轮结束，自动开始新一轮
      setIsProcessing(true);
      const nextStarterId = roundEndResult.nextStarter;
      const lastActualPlay = [...playedCards].reverse().find(p => p.cards.length > 0);
      const lastPlayerId = lastActualPlay?.playerId ?? 0;
      const lastPlayer = players.find(p => p.id === lastPlayerId);

      // 下一轮首发
      setRoundStarter(nextStarterId);
      setPlayedCards([]); // 清空出牌记录，开始新一轮
      setPlayerShowCards({}); // 清空玩家面前展示的牌
      setCurrentPlayer(nextStarterId);
      setCurrentRoundCardCount(null);
      setFirstPlay(true); // 新一轮，标记为首次出牌

      // 显示消息：如果是接风则特殊提示
      if (lastPlayer?.isOut && lastPlayerId !== nextStarterId) {
        showMessage(`一轮结束！${lastPlayer.name} 出完牌，${players[nextStarterId]?.name} 接风出牌`, 'info');
      } else {
        showMessage(`一轮结束！${players[nextStarterId]?.name} 最大，下一轮从ta开始`, 'info');
      }
      setIsProcessing(false);
      return;
    }

    const timer = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          // 时间到
          if (currentPlayer === 0 && !isProcessing) {
            setIsProcessing(true);

            // 首轮首发必须出牌，自动出最右边的一张
            if (firstPlay && playedCards.length === 0) {
              // 自动选择最右边的一张牌
              if (sortedCards.length > 0) {
                const autoCard = sortedCards[sortedCards.length - 1];
                setSelectedCards([autoCard]);

                // 模拟点击出牌
                setTimeout(() => {
                  // 调用出牌逻辑
                  const cardsToPlay = [autoCard];
                  const cardType = recognizeCardTypeFn(cardsToPlay, jokers, trumpRank);

                  if (cardType) {
                    setSortedCards(prev => prev.filter(c => c.id !== autoCard.id));
                    setPlayedCards(prev => [...prev, { playerId: 0, cards: cardsToPlay }]);
                    setPlayerShowCards(prev => ({ ...prev, 0: cardsToPlay }));
                    setFirstPlay(false);
                    setCurrentRoundCardCount(1); // 记录首轮出牌数量
                    showMessage(`超时自动出牌: ${getCardDisplay(autoCard)}`, 'info');

                    // 检查是否获胜
                    if (sortedCards.length - 1 === 0) {
                      setRoundWinner(0);
                      setGamePhase(GamePhase.ROUND_END);
                      showMessage('恭喜！你赢了！', 'success');
                      return;
                    }

                    // 继续下一个玩家
                    setCurrentPlayer((currentPlayer + 1) % 6);
                  }
                  setIsProcessing(false);
                }, 100);
                return 15;
              }
            } else {
              // 非首轮首发，时间到自动不出
              // 先构建更新后的 playedCards 用于检测
              const newPlayedCards = [...playedCards, { playerId: 0, cards: [] }];
              setPlayedCards(newPlayedCards);
              showMessage('时间到，你选择不出', 'info');

              // 检查是否一轮结束（使用更新后的 playedCards）
              const lastActualPlay = newPlayedCards.reverse().find(p => p.cards.length > 0);
              if (lastActualPlay && lastActualPlay.playerId === 0) {
                // 一轮结束，玩家是最后出牌者
                const lastPlayer = players.find(p => p.id === 0);

                if (lastPlayer?.isOut) {
                  // 玩家已出完，由下家接风
                  const nextActive = findNextActivePlayer(0, players);
                  showMessage(`一轮结束！你出完牌，${players[nextActive]?.name} 接风出牌`, 'info');
                  setRoundStarter(nextActive);
                  setPlayedCards([]);
                  setPlayerShowCards({});
                  setCurrentRoundCardCount(null);
                  setFirstPlay(true);
                  setCurrentPlayer(nextActive);
                } else {
                  showMessage(`一轮结束！你最大，下一轮从你开始`, 'info');
                  setRoundStarter(0);
                  setPlayedCards([]);
                  setPlayerShowCards({});
                  setCurrentRoundCardCount(null);
                  setFirstPlay(true);
                  setCurrentPlayer(0);
                }
                setIsProcessing(false);
                return 15;
              }

              // 继续下一个玩家
              const nextPlayer = findNextActivePlayer(0, players);
              setCurrentPlayer(nextPlayer);
              setIsProcessing(false);
            }
          }
          return 15;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [gamePhase, currentPlayer, isProcessing, playedCards.length, showMessage, firstPlay, sortedCards, jokers]);

  // AI出牌
  useEffect(() => {
    // 如果正在处理出牌，跳过
    if (isProcessing || gamePhase !== GamePhase.PLAYING || currentPlayer === 0) return;

    const aiPlayer = players[currentPlayer];
    if (!aiPlayer || aiPlayer.isOut) {
      // 找下一个有效玩家
      const nextActive = findNextActivePlayer(currentPlayer, players);
      setCurrentPlayer(nextActive);
      return;
    }

    // 标记开始处理
    setIsProcessing(true);

    // 检查是否一轮结束（牌权回到上一个出牌者）
    const roundEndResult = checkRoundEnd(currentPlayer, players);
    if (roundEndResult.isRoundEnd && roundEndResult.nextStarter !== null) {
      // 一轮结束，自动开始新一轮
      const nextStarterId = roundEndResult.nextStarter;
      const lastActualPlay = [...playedCards].reverse().find(p => p.cards.length > 0);
      const lastPlayerId = lastActualPlay?.playerId ?? 0;
      const lastPlayer = players.find(p => p.id === lastPlayerId);

      // 显示消息：如果是接风则特殊提示
      if (lastPlayer?.isOut && lastPlayerId !== nextStarterId) {
        showMessage(`一轮结束！${lastPlayer.name} 出完牌，${players[nextStarterId]?.name} 接风出牌`, 'info');
      } else {
        showMessage(`一轮结束！${players[nextStarterId]?.name} 最大，下一轮从ta开始`, 'info');
      }

      // 下一轮首发
      setRoundStarter(nextStarterId);
      setPlayedCards([]); // 清空出牌记录，开始新一轮
      setPlayerShowCards({}); // 清空玩家面前展示的牌
      setCurrentRoundCardCount(null);
      setFirstPlay(true); // 新一轮，标记为首次出牌
      setCurrentPlayer(nextStarterId); // 始终更新 currentPlayer，确保状态变化触发重新渲染
      setIsProcessing(false);
      return;
    }

    const timer = setTimeout(() => {
      // 找最后一个实际出牌的记录（不是"不出"）
      const lastActualPlay = playedCards.length > 0
        ? [...playedCards].reverse().find(p => p.cards.length > 0)
        : null;
      const currentPlay = lastActualPlay ? lastActualPlay.cards : null;
      
      // 检查是否需要出特定数量的牌
      let aiCards: Card[] | null = null;
      if (currentRoundCardCount !== null) {
        // 需要出特定数量，优先选择匹配数量的牌型
        aiCards = aiSelectCardsFn(aiPlayer.hand, jokers, currentPlay, firstPlay, currentRoundCardCount, trumpRank);
      } else {
        aiCards = aiSelectCardsFn(aiPlayer.hand, jokers, currentPlay, firstPlay, undefined, trumpRank);
      }
      
      if (aiCards && aiCards.length > 0) {
        // 记录首家出牌数量
        if (currentRoundCardCount === null) {
          setCurrentRoundCardCount(aiCards.length);
        }

        // 如果是首轮首发，标记 firstPlay 为 false
        if (firstPlay) {
          setFirstPlay(false);
        }

        // 检查是否出完牌
        const isAiOut = aiPlayer.hand.length - aiCards!.length === 0;

        setPlayers(prev => prev.map(p => {
          if (p.id === currentPlayer) {
            return {
              ...p,
              hand: p.hand.filter(c => !aiCards!.some(s => s.id === c.id)),
              cardCount: p.hand.length - aiCards!.length,
              isOut: isAiOut
            };
          }
          return p;
        }));

        setPlayedCards(prev => [...prev, { playerId: currentPlayer, cards: aiCards! }]);
        // 更新该AI玩家面前展示的牌
        setPlayerShowCards(prev => ({ ...prev, [currentPlayer]: aiCards! }));

        if (isAiOut) {
          // AI 出完牌，显示排名
          const rankName = getRankName(finishOrder.length);
          showMessage(`${aiPlayer.name} 出完牌，获得 ${rankName}`, 'success');

          // 记录排名
          setFinishOrder(prev => {
            const newOrder = [...prev, currentPlayer];

            // 检查是否所有人都出完牌
            if (newOrder.length === 6) {
              // 本局结束，检查是否需要进贡
              setTimeout(() => {
                const updatedPlayersList = players.map(p =>
                  p.id === currentPlayer ? { ...p, isOut: true, hand: p.hand.filter(c => !aiCards!.some(s => s.id === c.id)) } : p
                );
                handleGameEnd(newOrder, updatedPlayersList);
              }, 500);
            }

            return newOrder;
          });
        } else {
          showMessage(`${aiPlayer.name} 出了 ${aiCards!.map(c => getCardDisplay(c)).join(' ')}`, 'info');
        }
      } else {
        setPlayedCards(prev => [...prev, { playerId: currentPlayer, cards: [] }]);
        showMessage(`${aiPlayer.name} 不出`, 'info');
      }

      // 找到下一个未出完的玩家
      setPlayers(prevPlayers => {
        const isAiOut = aiCards && aiCards.length > 0 && aiPlayer.hand.length - aiCards.length === 0;
        const updatedPlayers = prevPlayers.map(p => {
          if (p.id === currentPlayer && isAiOut) {
            return { ...p, isOut: true };
          }
          return p;
        });

        const nextPlayer = findNextActivePlayer(currentPlayer, updatedPlayers);
        setCurrentPlayer(nextPlayer);
        setIsProcessing(false);

        return updatedPlayers;
      });
    }, 800);

    return () => clearTimeout(timer);
  }, [gamePhase, currentPlayer, players, playedCards, jokers, firstPlay, showMessage, roundStarter, checkRoundEnd, calculateScore, findNextActivePlayer, finishOrder.length, trumpRank]);

  // 开始下一盘（保留分数，更新庄家和首发）
  const startNextRound = useCallback(() => {
    // 重置防重入守卫，允许新一盘正常结算
    gameEndCalledRef.current = false;
    aiReturnProcessedRef.current = '';

    // 发新牌
    const deck = createDeckFn();
    const shuffled = shuffleDeckFn(deck);
    const hands = dealCardsFn(shuffled, 6);

    const jokerCards = shuffled.filter(c => c.isJoker);
    setJokers(jokerCards);

    // 玩家手牌按大小排序
    const sortedHand = sortCardsFn(hands[0], jokerCards);

    const newPlayers: Player[] = Array.from({ length: 6 }, (_, i) => ({
      id: i,
      name: PLAYER_NAMES[i],
      hand: i === 0 ? sortedHand : sortCardsFn(hands[i], jokerCards),
      position: i,
      team: i % 2 === 0 ? 1 : 2,
      isAI: true,
      isOut: false,
      cardCount: hands[i].length,
      hasAsked: false
    }));

    // 初始化理牌区
    setSortedCards([...sortedHand]);

    setPlayers(newPlayers);
    setSelectedCards([]);
    setGamePhase(GamePhase.TEAMING);
    setPlayedCards([]);
    setPlayerShowCards({});
    setRoundWinner(null);
    setFirstPlay(true);
    setCurrentRoundCardCount(null);
    setTrumpRank(null);
    setFinishOrder([]);
    setLastTributeInfo([]); // 清空上盘进贡信息

    // 设置首发（头游选手）
    const firstPlaceId = finishOrder.length > 0 ? finishOrder[0] : null;
    if (firstPlaceId !== null) {
      setRoundStarter(firstPlaceId);
      setCurrentPlayer(firstPlaceId);
    }

    const starterId = firstPlaceId ?? 0;

    // 检查是否有待执行的进贡
    if (pendingTributeInfo && pendingTributeInfo.tributors.length > 0) {
      // 有待执行的进贡，进入进贡阶段（进贡完成后再自动设将牌）
      setTributeInfo(pendingTributeInfo);
      setPendingTributeInfo(null);
      setGamePhase(GamePhase.TRIBUTE);

      const tributorNames = pendingTributeInfo.tributors.map(id => newPlayers.find(p => p.id === id)?.name).join('、');
      showMessage(`进贡阶段！${tributorNames} 需要向头游方进贡`, 'info');
    } else if (!isFirstRound) {
      // 非第一盘且无进贡：根据庄家方等级自动设将牌
      // 注意：dealerTeam 此时已由 handleGameEnd 更新为本盘赢家方
      const autoRank = dealerTeam === 1 ? redLevel : blueLevel;
      // 对新发的牌直接排序（此时 state 中的 players/jokers 还未更新，传入 newPlayers/jokerCards）
      const sortedByTrump = newPlayers.map(p => ({
        ...p,
        hand: sortCardsFn(p.hand, jokerCards, autoRank)
      }));
      setPlayers(sortedByTrump);
      setSortedCards(sortCardsFn(sortedHand, jokerCards, autoRank));
      setTrumpRank(autoRank);

      const trumpName = TRUMP_OPTIONS.find(t => t.rank === autoRank)?.name || String(autoRank);
      showMessage(`本局${dealerTeam === 1 ? '红方' : '蓝方'}打${trumpName}，自动设将牌`, 'info');

      setTimeout(() => {
        setGamePhase(GamePhase.PLAYING);
        setCurrentPlayer(starterId);
        setRoundStarter(starterId);
        const starterName = newPlayers.find(p => p.id === starterId)?.name || '一号';
        showMessage(`${starterName}先出牌`, 'info');
      }, 1000);
    } else {
      // 第一盘：进入手选将牌阶段
      setGamePhase(GamePhase.TEAMING);
      showMessage('请选择本局将牌...', 'info');
    }
  }, [finishOrder, players, pendingTributeInfo, isFirstRound, dealerTeam, redLevel, blueLevel, showMessage]);

  // 渲染卡牌
  const renderCard = (card: Card, isSelected: boolean = false) => {
    const isRed = isRedSuit(card.suit);
    const isJoker = card.isJoker;
    
    const getRankDisplay = (rank: number): string => {
      const map: Record<number, string> = {
        3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: '10', 11: 'J', 12: 'Q', 13: 'K', 14: 'A', 15: '2'
      };
      return map[rank] || String(rank);
    };
    
    let displayText = '';
    let suitDisplay = '';
    
    if (card.isBigJoker || card.isSmallJoker) {
      displayText = '';
      suitDisplay = '🃏';
    } else {
      displayText = getRankDisplay(card.rank as number);
      suitDisplay = card.suit;
    }

    return (
      <div
        key={card.id}
        className={`card ${isRed ? 'red' : 'black'} ${isJoker ? 'joker' : ''} ${isSelected ? 'selected' : ''} ${card.isBigJoker ? 'big-joker' : ''} ${card.isSmallJoker ? 'small-joker' : ''}`}
      >
        <div className="card-inner">
          <div className="corner top">
            <span>{displayText}</span>
            <span>{suitDisplay}</span>
          </div>
          <div className="center">
            {card.isBigJoker && <span className="joker-center">🃏</span>}
            {card.isSmallJoker && <span className="joker-center">🃏</span>}
            {!card.isJoker && suitDisplay}
          </div>
          <div className="corner bottom">
            <span>{displayText}</span>
            <span>{suitDisplay}</span>
          </div>
        </div>
      </div>
    );
  };

  // 渲染理牌区的牌（均匀叠放，每张牌都露出点数和花色）
  const renderSortedCards = () => {
    // 每张牌只向右偏移10px，这样所有牌都叠在一起，每张都能看到
    const cardOffset = 10;
    const totalWidth = sortedCards.length > 1 
      ? 48 + (sortedCards.length - 1) * cardOffset 
      : 60;
    
    return (
      <div className="sort-cards-container" style={{ width: `${totalWidth}px` }}>
        {sortedCards.map((card, index) => {
          const isSelected = selectedCards.some(c => c.id === card.id);
          const isDragging = draggingIndex === index;
          const isMultiDragging = draggingIndices.includes(index); // 是否在多选拖动中
          const isDragOver = dragOverIndex.current === index;
          
          return (
            <div 
              key={card.id}
              className="card-wrapper-stacked"
              draggable={gamePhase === GamePhase.PLAYING}
              onDragStart={() => handleDragStart(index)}
              onDragOver={(e) => handleDragOver(e, index)}
              onDragEnd={handleDragEnd}
              onClick={() => toggleCardSelection(card)}
              style={{
                marginLeft: index > 0 ? '-10px' : '0',
                zIndex: isDragging || isMultiDragging ? 100 : (isDragOver ? 50 : index),
                opacity: isMultiDragging ? 0.7 : 1,
              }}
            >
              <div className={`card ${isSelected ? 'selected' : ''} ${isDragging || isMultiDragging ? 'dragging' : ''} ${isDragOver ? 'drag-over' : ''} ${isRedSuit(card.suit) ? 'red' : 'black'} ${card.isBigJoker ? 'big-joker' : ''} ${card.isSmallJoker ? 'small-joker' : ''}`}>
                <div className="card-inner">
                  <div className="corner top">
                    <span>{card.isBigJoker || card.isSmallJoker ? '' : 
                      (() => {
                        const map: Record<number, string> = {3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'J',12:'Q',13:'K',14:'A',15:'2'};
                        return map[card.rank as number] || String(card.rank);
                      })()
                    }</span>
                    <span>{card.isJoker ? '🃏' : card.suit}</span>
                  </div>
                  <div className="center">
                    {card.isBigJoker && <span>🃏</span>}
                    {card.isSmallJoker && <span>🃏</span>}
                    {!card.isJoker && card.suit}
                  </div>
                  <div className="corner bottom">
                    <span>{card.isBigJoker || card.isSmallJoker ? '' : 
                      (() => {
                        const map: Record<number, string> = {3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'J',12:'Q',13:'K',14:'A',15:'2'};
                        return map[card.rank as number] || String(card.rank);
                      })()
                    }</span>
                    <span>{card.isJoker ? '🃏' : card.suit}</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // 渲染玩家座位
  const renderPlayerSeat = (player: Player) => {
    const isActive = currentPlayer === player.id && gamePhase === GamePhase.PLAYING;
    const showCards = playerShowCards[player.id] || []; // 获取该玩家当前展示的牌
    const playerRank = finishOrder.indexOf(player.id); // 玩家排名，-1表示未出完

    // 左右两边的玩家（1、2、4、5号）使用横向布局
    const isHorizontal = [1, 2, 4, 5].includes(player.id);

    // 如果玩家已出完牌，显示排名和最后出的牌
    if (player.isOut && playerRank >= 0) {
      const rankName = getRankName(playerRank);
      return (
        <div key={player.id} className={`player-seat player-seat-${player.id} player-finished`}>
          <div className="player-info-wrapper">
            <div className={`player-avatar team${player.team}`}>
              {AVATARS[player.id]}
            </div>
            <div className={`player-info team${player.team}`}>
              <div className="name">{player.name}</div>
              <div className="rank-badge">{rankName}</div>
              <div className="team-badge">队{player.team}</div>
            </div>
          </div>
          {/* 出完牌的玩家也要显示最后出的牌 */}
          {showCards.length > 0 && (
            <div className="player-played">
              <div style={{
                display: 'flex',
                alignItems: 'flex-end',
                minWidth: showCards.length > 1 ? 48 + (showCards.length - 1) * 10 : 50
              }}>
                {showCards.map((card, index) => (
                  <div
                    key={card.id}
                    style={{
                      marginLeft: index > 0 ? '-10px' : '0',
                      zIndex: index
                    }}
                  >
                    {renderCard(card)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      );
    }

    if (isHorizontal) {
      // 横向布局：出牌区在内侧（桌子中心方向）
      const isRight = [1, 2].includes(player.id); // 右侧玩家

      return (
        <div key={player.id} className={`player-seat player-seat-${player.id}`}>
          {isRight ? (
            // 右侧：头像和信息在外（右），出牌区在内（左）
            <>
              <div className="player-info-wrapper">
                <div className={`player-avatar team${player.team}`}>
                  {AVATARS[player.id]}
                </div>
                <div className={`player-info ${isActive ? 'active' : ''} team${player.team}`}>
                  <div className="name">{player.name}</div>
                  <div className="cards-left">{player.cardCount}张</div>
                  <div className="team-badge">队{player.team}</div>
                </div>
              </div>
              <div className="player-played">
                {showCards.length > 0 ? (
                  <div style={{
                    display: 'flex',
                    alignItems: 'flex-end',
                    minWidth: showCards.length > 1 ? 48 + (showCards.length - 1) * 10 : 50
                  }}>
                    {showCards.map((card, index) => (
                      <div
                        key={card.id}
                        style={{
                          marginLeft: index > 0 ? '-10px' : '0',
                          zIndex: index
                        }}
                      >
                        {renderCard(card)}
                      </div>
                    ))}
                  </div>
                ) : (
                  <span style={{ color: '#888', fontSize: '0.6rem' }}>-</span>
                )}
              </div>
            </>
          ) : (
            // 左侧：出牌区在内（右），头像和信息在外（左）
            <>
              <div className="player-played">
                {showCards.length > 0 ? (
                  <div style={{
                    display: 'flex',
                    alignItems: 'flex-end',
                    minWidth: showCards.length > 1 ? 48 + (showCards.length - 1) * 10 : 50
                  }}>
                    {showCards.map((card, index) => (
                      <div
                        key={card.id}
                        style={{
                          marginLeft: index > 0 ? '-10px' : '0',
                          zIndex: index
                        }}
                      >
                        {renderCard(card)}
                      </div>
                    ))}
                  </div>
                ) : (
                  <span style={{ color: '#888', fontSize: '0.6rem' }}>-</span>
                )}
              </div>
              <div className="player-info-wrapper">
                <div className={`player-avatar team${player.team}`}>
                  {AVATARS[player.id]}
                </div>
                <div className={`player-info ${isActive ? 'active' : ''} team${player.team}`}>
                  <div className="name">{player.name}</div>
                  <div className="cards-left">{player.cardCount}张</div>
                  <div className="team-badge">队{player.team}</div>
                </div>
              </div>
            </>
          )}
        </div>
      );
    }

    // 上下玩家（0、3号）使用纵向布局
    return (
      <div key={player.id} className={`player-seat player-seat-${player.id}`}>
        <div className={`player-avatar team${player.team}`}>
          {AVATARS[player.id]}
        </div>
        <div className={`player-info ${isActive ? 'active' : ''} team${player.team}`}>
          <div className="name">{player.name}</div>
          <div className="cards-left">{player.cardCount}张</div>
          <div className="team-badge">队{player.team}</div>
        </div>
        <div className="player-played">
          {showCards.length > 0 ? (
            <div style={{
              display: 'flex',
              alignItems: 'flex-end',
              minWidth: showCards.length > 1 ? 48 + (showCards.length - 1) * 10 : 50
            }}>
              {showCards.map((card, index) => (
                <div
                  key={card.id}
                  style={{
                    marginLeft: index > 0 ? '-10px' : '0',
                    zIndex: index
                  }}
                >
                  {renderCard(card)}
                </div>
              ))}
            </div>
          ) : (
            <span style={{ color: '#888', fontSize: '0.6rem' }}>等待出牌</span>
          )}
        </div>
      </div>
    );
  };

  // 开始界面
  if (gamePhase === GamePhase.DEALING) {
    return (
      <div className="app">
        <div className="start-screen">
          <h1>大怪路子</h1>
          <p className="subtitle-text">
            上海经典六人纸牌游戏<br/>
            3V3组队对战，牌型丰富有趣
          </p>
          <button className="btn btn-primary" onClick={initGame}>
            开始游戏
          </button>
        </div>
      </div>
    );
  }

  // 进贡阶段界面
  if (gamePhase === GamePhase.TRIBUTE && tributeInfo) {
    const currentTributorId = tributeInfo.tributors[tributeInfo.currentTributorIndex];
    const currentTributor = players.find(p => p.id === currentTributorId);
    const isPlayerTributing = currentTributorId === 0;

    // 检查当前进贡方是否可以抗贡
    const canResist = isPlayerTributing && countJokers(0, players) >= 3;
    
    // 获取玩家手中最大的牌（可能有多张同点数不同花色）
    const bestCards = isPlayerTributing ? getBestCards(0, players) : [];

    return (
      <div className="phase-overlay">
        <h2>进贡阶段</h2>
        <div className="tribute-info">
          <p className="tribute-status">
            {tributeInfo.tributors.map((id, idx) => {
              const p = players.find(pl => pl.id === id);
              const isResisted = tributeInfo.resistedPlayers.includes(id);
              const hasTributed = tributeInfo.tributes.some(t => t.tributorId === id);
              const isCurrent = idx === tributeInfo.currentTributorIndex;

              return (
                <span key={id} className={`tribute-player ${isCurrent ? 'current' : ''} ${isResisted ? 'resisted' : ''} ${hasTributed ? 'done' : ''}`}>
                  {p?.name}
                  {isResisted && ' (抗贡)'}
                  {hasTributed && !isResisted && ' (已进贡)'}
                  {isCurrent && !isResisted && !hasTributed && ' (进贡中)'}
                </span>
              );
            })}
          </p>
        </div>

        {isPlayerTributing && !canResist && (
          <div className="tribute-selection">
            <h3>请选择要进贡的牌（必须进贡最大的牌）</h3>
            {bestCards.length > 1 && (
              <p className="tribute-hint">你有 {bestCards.length} 张最大的牌，请选择一张进贡</p>
            )}
            <div className="tribute-countdown">
              剩余时间: <span className={tributeCountdown <= 5 ? 'countdown-warning' : ''}>{tributeCountdown}秒</span>
            </div>
            <div className="tribute-cards">
              {bestCards.map(card => (
                <div
                  key={card.id}
                  className={`tribute-card ${selectedTributeCard?.id === card.id ? 'selected' : ''}`}
                  onClick={() => handleTributeSelect(card)}
                >
                  {renderCard(card)}
                </div>
              ))}
            </div>
            <button
              className="btn btn-primary"
              onClick={confirmTribute}
              disabled={!selectedTributeCard}
            >
              确认进贡
            </button>
          </div>
        )}

        {isPlayerTributing && canResist && !tributeForceNoResist && (
          <div className="tribute-resist">
            <h3>你有 {countJokers(0, players)} 个王，可以抗贡！</h3>
            {/* 展示玩家的王 */}
            <div className="joker-display">
              <div className="joker-cards">
                {getJokerCards(0, players).map(card => (
                  <div key={card.id} className="joker-card">
                    {renderCard(card)}
                  </div>
                ))}
              </div>
            </div>
            <button
              className="btn btn-primary"
              onClick={confirmResist}
            >
              抗贡
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => {
                // 选择进贡而非抗贡
                setTributeForceNoResist(true);
                setTributeCountdown(15);
              }}
            >
              放弃抗贡，选择进贡
            </button>
          </div>
        )}

        {/* 玩家放弃抗贡后，选择进贡 */}
        {isPlayerTributing && canResist && tributeForceNoResist && (
          <div className="tribute-selection">
            <h3>请选择要进贡的牌（必须进贡最大的牌）</h3>
            {bestCards.length > 1 && (
              <p className="tribute-hint">你有 {bestCards.length} 张最大的牌，请选择一张进贡</p>
            )}
            <div className="tribute-countdown">
              剩余时间: <span className={tributeCountdown <= 5 ? 'countdown-warning' : ''}>{tributeCountdown}秒</span>
            </div>
            <div className="tribute-cards">
              {bestCards.map(card => (
                <div
                  key={card.id}
                  className={`tribute-card ${selectedTributeCard?.id === card.id ? 'selected' : ''}`}
                  onClick={() => handleTributeSelect(card)}
                >
                  {renderCard(card)}
                </div>
              ))}
            </div>
            <button
              className="btn btn-primary"
              onClick={confirmTribute}
              disabled={!selectedTributeCard}
            >
              确认进贡
            </button>
          </div>
        )}

        {!isPlayerTributing && (
          <div className="tribute-waiting">
            <p>等待 {currentTributor?.name} 进贡...</p>
          </div>
        )}

        {/* 抗贡成功后展示王的界面（不论玩家是否是当前进贡方） */}
        {tributeInfo.jokerCards.length > 0 && (
          <div className="joker-display">
            <p className="joker-display-title">抗贡展示 {tributeInfo.jokerCards.length} 个王！</p>
            <div className="joker-cards">
              {tributeInfo.jokerCards.map(card => (
                <div key={card.id} className="joker-card">
                  {renderCard(card)}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  // 还贡阶段界面
  if (gamePhase === GamePhase.TRIBUTE_RETURN && tributeInfo) {
    const validTributes = tributeInfo.tributes.filter(t =>
      !tributeInfo.resistedPlayers.includes(t.tributorId)
    );

    if (tributeInfo.currentTributorIndex >= validTributes.length) {
      return null;
    }

    const currentTribute = validTributes[tributeInfo.currentTributorIndex];
    const receiverId = currentTribute.receiverId;
    const tributorId = currentTribute.tributorId;
    const receiver = players.find(p => p.id === receiverId);
    const tributor = players.find(p => p.id === tributorId);
    const needed = getReturnCandidateCount(currentTribute.card);

    // 进贡牌说明
    const tributeCardDesc = currentTribute.card.isBigJoker
      ? '大王'
      : currentTribute.card.isSmallJoker
      ? '小王'
      : getCardDisplay(currentTribute.card);

    // ── 阶段A：受贡方选候选牌 ─────────────────────────────────────────
    if (tributeInfo.returnSubPhase === 'selecting_candidates') {
      const isPlayerReceiver = receiverId === 0;
      const allOptions = getReturnCardOptions(receiverId, players);

      // 还贡规则说明
      const ruleText = needed === 1
        ? `收到非王牌（${tributeCardDesc}），请直接选 1 张牌还贡`
        : needed === 2
        ? `收到小王，请选 2 张候选牌展示给 ${tributor?.name} 挑选`
        : `收到大王，请选 3 张候选牌展示给 ${tributor?.name} 挑选`;

      return (
        <div className="phase-overlay">
          <h2>还贡阶段</h2>
          <div className="tribute-received">
            <p>{receiver?.name} 收到了来自 {tributor?.name} 的进贡：</p>
            <div className="tribute-card-received">
              {renderCard(currentTribute.card)}
              <span className="tribute-card-name">{tributeCardDesc}</span>
            </div>
            <p className="return-rule">{ruleText}</p>
          </div>

          {isPlayerReceiver && allOptions.length > 0 && (
            <div className="tribute-selection">
              <div className={`tribute-countdown ${returnCountdown <= 5 ? 'countdown-warning' : ''}`}>
                倒计时: {returnCountdown}秒
              </div>
              <h3>
                请选择 {needed} 张候选牌
                {selectedReturnCandidates.length > 0 && (
                  <span style={{ marginLeft: 8, color: '#4caf50' }}>
                    已选 {selectedReturnCandidates.length}/{needed}
                  </span>
                )}
              </h3>
              <div className="tribute-cards">
                {allOptions.map(card => {
                  const isSelected = selectedReturnCandidates.some(c => c.id === card.id);
                  return (
                    <div
                      key={card.id}
                      className={`tribute-card ${isSelected ? 'selected' : ''}`}
                      onClick={() => handleReturnCandidateToggle(card)}
                    >
                      {renderCard(card)}
                    </div>
                  );
                })}
              </div>
              <button
                className="btn btn-primary"
                onClick={confirmReturnCandidates}
                disabled={selectedReturnCandidates.length !== needed}
              >
                {needed === 1 ? '确认还贡' : `确认展示 ${needed} 张候选`}
              </button>
            </div>
          )}

          {!isPlayerReceiver && (
            <div className="tribute-waiting">
              <p>等待 {receiver?.name} 选择候选牌...</p>
            </div>
          )}
        </div>
      );
    }

    // ── 阶段B：进贡方从候选中挑1张 ──────────────────────────────────
    if (tributeInfo.returnSubPhase === 'tributor_picking') {
      const isPlayerTributor = tributorId === 0;
      const candidates = tributeInfo.returnCandidates;

      return (
        <div className="phase-overlay">
          <h2>还贡阶段 · 挑选</h2>
          <div className="tribute-received">
            <p>{receiver?.name} 展示了 {candidates.length} 张候选牌，请 {tributor?.name} 挑选 1 张：</p>
          </div>

          {isPlayerTributor && candidates.length > 0 && (
            <div className="tribute-selection">
              <div className={`tribute-countdown ${returnCountdown <= 5 ? 'countdown-warning' : ''}`}>
                倒计时: {returnCountdown}秒
              </div>
              <h3>请从候选牌中选择 1 张</h3>
              <div className="tribute-cards">
                {candidates.map(card => (
                  <div
                    key={card.id}
                    className={`tribute-card ${selectedReturnCard?.id === card.id ? 'selected' : ''}`}
                    onClick={() => handleReturnSelect(card)}
                  >
                    {renderCard(card)}
                  </div>
                ))}
              </div>
              <button
                className="btn btn-primary"
                onClick={confirmReturn}
                disabled={!selectedReturnCard}
              >
                确认挑牌
              </button>
            </div>
          )}

          {!isPlayerTributor && (
            <div className="tribute-waiting">
              <p>等待 {tributor?.name} 从候选牌中挑选...</p>
              <div className="tribute-cards" style={{ pointerEvents: 'none', opacity: 0.7 }}>
                {candidates.map(card => (
                  <div key={card.id} className="tribute-card">
                    {renderCard(card)}
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

  // 回合结束界面
  if (gamePhase === GamePhase.ROUND_END && roundWinner !== null) {
    const { winnerTeam, score } = finishOrder.length === 6 ? calculateScore(finishOrder, players) : { winnerTeam: null, score: 0 };
    const firstPlacePlayer = finishOrder.length > 0 ? players.find(p => p.id === finishOrder[0]) : null;
    // 计算下盘双方的打几（赢家方若得分>0则升一级）
    const nextRedLevel = (winnerTeam === 1 && score > 0) ? Math.min(redLevel + 1, 14) : redLevel;
    const nextBlueLevel = (winnerTeam === 2 && score > 0) ? Math.min(blueLevel + 1, 14) : blueLevel;
    const nextRedName = TRUMP_OPTIONS.find(t => t.rank === nextRedLevel)?.name || String(nextRedLevel);
    const nextBlueName = TRUMP_OPTIONS.find(t => t.rank === nextBlueLevel)?.name || String(nextBlueLevel);
    const nextDealerTeam = winnerTeam ?? dealerTeam;

    return (
      <div className="phase-overlay">
        <h2>本局结束</h2>
        <div className="finish-ranking">
          <h3>排名</h3>
          {finishOrder.map((playerId, index) => {
            const p = players.find(pl => pl.id === playerId);
            if (!p) return null;
            return (
              <div key={playerId} className={`rank-item team${p.team}`}>
                <span className="rank-num">{getRankName(index)}</span>
                <span className="rank-name">{p.name}</span>
                <span className={`rank-team team${p.team}`}>{p.team === 1 ? '红队' : '蓝队'}</span>
              </div>
            );
          })}
        </div>
        <div className="score-result">
          {winnerTeam === 1 ? (
            score > 0 ? (
              <p className="score-text success">红队获胜！得 {score} 分</p>
            ) : (
              <p className="score-text">红队头游，但不得分（末游是红方）</p>
            )
          ) : winnerTeam === 2 ? (
            score > 0 ? (
              <p className="score-text success">蓝队获胜！得 {score} 分</p>
            ) : (
              <p className="score-text">蓝队头游，但不得分（末游是蓝方）</p>
            )
          ) : null}
        </div>
        <div className="total-scores">
          <span className="red-score">红队: {scores.red}分</span>
          <span className="blue-score">蓝队: {scores.blue}分</span>
        </div>
        <div className="next-round-info">
          <p>下一盘庄家: {nextDealerTeam === 1 ? '红队' : '蓝队'}</p>
          <p>下一盘首发: {firstPlacePlayer?.name || '-'}</p>
          <p className="next-level-info">
            下盘将牌: <span className="team-red">红打{nextRedName}</span>
            {' / '}
            <span className="team-blue">蓝打{nextBlueName}</span>
            {'（由'}{nextDealerTeam === 1 ? <span className="team-red">红方</span> : <span className="team-blue">蓝方</span>}{'作庄，打'}{nextDealerTeam === 1 ? <span className="team-red">{nextRedName}</span> : <span className="team-blue">{nextBlueName}</span>}{'）'}
          </p>
        </div>
        <button className="btn btn-primary" onClick={startNextRound}>
          下一盘
        </button>
      </div>
    );
  }

  return (
    <div className="app">
      {/* 将牌选择界面 */}
      {gamePhase === GamePhase.TEAMING && trumpRank === null && (
        <div className="trump-selection-overlay">
          <div className="trump-selection-modal">
            <h2>选择本局将牌</h2>
            <p>将牌比所有普通牌都大，但比大小王小</p>
            <div className="trump-options">
              {TRUMP_OPTIONS.map(option => (
                <button
                  key={option.rank}
                  className="trump-btn"
                  onClick={() => selectTrump(option.rank)}
                >
                  打{option.name}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      <header className="header">
        <h1>大怪路子</h1>
        <div className="subtitle">
          {gamePhase === GamePhase.TEAMING && trumpRank === null
            ? '选择将牌中...'
            : gamePhase === GamePhase.TEAMING
              ? '组队中...'
              : `游戏中 - ${roundStarter !== null ? players[roundStarter]?.name + '开牌' : ''} 逆时针 | 将牌: ${TRUMP_OPTIONS.find(t => t.rank === trumpRank)?.name || '-'}`}
        </div>
      </header>

      {/* 右上角游戏状态 */}
      {gamePhase === GamePhase.PLAYING && trumpRank && (
        <div className="game-status-panel">
          <div className="status-row">
            <span className="status-label">本局:</span>
            <span className="status-value trump">打{TRUMP_OPTIONS.find(t => t.rank === trumpRank)?.name}</span>
          </div>
          <div className="status-row">
            <span className="status-label">庄家:</span>
            <span className={`status-value ${dealerTeam === 1 ? 'team-red' : 'team-blue'}`}>
              {dealerTeam === 1 ? '红方' : '蓝方'}
            </span>
          </div>
          <div className="score-row">
            <span className="red-score">红 {scores.red}</span>
            <span className="score-sep">:</span>
            <span className="blue-score">{scores.blue} 蓝</span>
          </div>
          <div className="level-row">
            <span className="red-level">红打{TRUMP_OPTIONS.find(t => t.rank === redLevel)?.name || redLevel}</span>
            <span className="level-sep"> / </span>
            <span className="blue-level">蓝打{TRUMP_OPTIONS.find(t => t.rank === blueLevel)?.name || blueLevel}</span>
          </div>
        </div>
      )}

      {/* 左上角进贡信息 */}
      {gamePhase === GamePhase.PLAYING && lastTributeInfo.length > 0 && (
        <div className="tribute-log-panel">
          <div className="tribute-log-title">本盘进贡</div>
          {lastTributeInfo.map((info, idx) => (
            <div key={idx} className="tribute-log-item">
              <span className="tribute-log-from">{info.tributorName}</span>
              <span className="tribute-log-arrow"> → </span>
              <span className="tribute-log-to">{info.receiverName}</span>
              <span className="tribute-log-card"> [{getCardDisplay(info.card)}]</span>
            </div>
          ))}
        </div>
      )}

      <div className={`message-bar ${message.type}`}>
        {message.text || '等待出牌...'}
      </div>

      <div className="game-area">
        <div className="table-surface">
          {players.map(renderPlayerSeat)}
          {/* 每个玩家面前都有自己的出牌区，不再显示中央出牌区 */}
        </div>
      </div>

      <div className="bottom-controls">
        {/* 理牌区 */}
        <div className="sort-area">
          <div className="sort-area-label">
            理牌区 ({sortedCards.length}张) - 点击选牌，拖拽排序，牌多自动叠放
          </div>
          {renderSortedCards()}
        </div>

        <div className="controls">
          {currentPlayer === 0 && gamePhase === GamePhase.PLAYING ? (
            <>
              <div style={{ color: countdown <= 5 ? '#e74c3c' : '#e8c547', fontSize: '0.9rem', marginRight: '10px' }}>
                {countdown}秒
              </div>
              <button 
                className="btn btn-primary" 
                onClick={playCards}
                disabled={selectedCards.length === 0}
              >
                出牌 ({selectedCards.length}张)
              </button>
              <button
                className="btn btn-secondary"
                onClick={passPlay}
                disabled={playedCards.length === 0 || (firstPlay && playedCards.length === 0)}
              >
                不出
              </button>
              <button 
                className="btn btn-secondary" 
                onClick={() => setSelectedCards([])}
                disabled={selectedCards.length === 0}
              >
                取消选择
              </button>
            </>
          ) : (
            <div style={{ color: '#8fbcAA', fontSize: '0.85rem' }}>
              请等待 {players[currentPlayer]?.name} 出牌...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
