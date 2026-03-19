// 花色
export const Suit = {
  SPADES: '♠',
  HEARTS: '♥',
  CLUBS: '♣',
  DIAMONDS: '♦',
  JOKER_RED: 'joker-red',
  JOKER_BLACK: 'joker-black'
} as const;
export type Suit = typeof Suit[keyof typeof Suit];

// 点数（注意：2作为普通牌时是最小的，值为2）
// 大怪路子规则：3>4>5>6>7>8>9>10>J>Q>K>A>2(最小)
// 但当某张牌成为"将牌"时，它就比其他普通牌都大
export const Rank = {
  '2': 2,   // 2最小
  '3': 3,
  '4': 4,
  '5': 5,
  '6': 6,
  '7': 7,
  '8': 8,
  '9': 9,
  '10': 10,
  'J': 11,
  'Q': 12,
  'K': 13,
  'A': 14,
  JOKER: 16  // 小王
} as const;
export type Rank = typeof Rank[keyof typeof Rank];

// 特殊点数
export const RANK_JOKER_BIG = 17 // 大王

// 牌型枚举
export enum CardType {
  SINGLE = 'single',           // 单张
  PAIR = 'pair',               // 对牌
  TRIPLE = 'triple',           // 三张
  STRAIGHT = 'straight',       // 杂顺
  FLUSH = 'flush',             // 同花
  THREE_WITH_TWO = 'three_with_two', // 三带两
  FOUR_WITH_ONE = 'four_with_one',    // 四带一
  STRAIGHT_FLUSH = 'straight_flush',  // 同花顺
  FIVE_OF_KIND = 'five_of_kind'       // 五同
}

// 牌型大小排序 (从大到小)
export const CARD_TYPE_ORDER: Record<CardType, number> = {
  [CardType.FIVE_OF_KIND]: 9,
  [CardType.STRAIGHT_FLUSH]: 8,
  [CardType.FOUR_WITH_ONE]: 7,
  [CardType.THREE_WITH_TWO]: 6,
  [CardType.FLUSH]: 5,
  [CardType.STRAIGHT]: 4,
  [CardType.TRIPLE]: 3,
  [CardType.PAIR]: 2,
  [CardType.SINGLE]: 1
}

// 牌接口
export interface Card {
  id: string;
  suit: Suit;
  rank: Rank | number;  // 可以是Rank枚举或大王(17)
  isJoker: boolean;
  isBigJoker: boolean;
  isSmallJoker: boolean;
}

// 玩家接口
export interface Player {
  id: number;
  name: string;
  hand: Card[];
  position: number;  // 1-6
  team: number;      // 1 或 2
  isAI: boolean;
  isOut: boolean;    // 是否已出完牌
  cardCount: number; // 剩余牌数
  hasAsked: boolean; // 本轮是否已问牌
}

// 游戏阶段
export enum GamePhase {
  DEALING = 'dealing',       // 发牌中
  TEAMING = 'teaming',       // 抽签组队中
  PLAYING = 'playing',       // 出牌中
  ROUND_END = 'round_end',   // 回合结束
  TRIBUTE = 'tribute',       // 进贡阶段
  TRIBUTE_RETURN = 'tribute_return', // 还贡阶段
  GAME_OVER = 'game_over'   // 游戏结束
}

// 还贡子阶段
export type ReturnSubPhase = 'selecting_candidates' | 'tributor_picking';

// 进贡信息
export interface TributeInfo {
  tributors: number[];       // 需要进贡的玩家ID列表（按出牌顺序倒序，末游在前）
  receivers: number[];       // 接受进贡的玩家ID列表（头游方，按出牌顺序正序，头游在前）
  currentTributorIndex: number; // 当前正在进贡的玩家索引
  currentReceiverIndex: number; // 当前接受进贡的玩家索引
  tributes: { tributorId: number; card: Card; receiverId: number }[]; // 已进贡的牌
  returns: { tributorId: number; card: Card; receiverId: number }[]; // 已还贡的牌
  canResist: boolean;        // 当前进贡方是否可以抗贡
  resistedPlayers: number[]; // 已抗贡的玩家ID列表
  jokerCards: Card[];        // 抗贡时展示的王
  // 还贡子阶段相关
  returnSubPhase: ReturnSubPhase; // 'selecting_candidates'=受贡方选候选 / 'tributor_picking'=进贡方挑牌
  returnCandidates: Card[];  // 受贡方已选出的候选牌（进贡方从中挑1张）
}

// 出牌记录
export interface PlayRecord {
  playerId: number;
  cards: Card[];
  cardType: CardType;
  timestamp: number;
}

// 消息类型
export interface GameMessage {
  id: number;
  text: string;
  type: 'info' | 'warning' | 'success' | 'error';
  timestamp: number;
}

// 获取牌面显示
export function getCardDisplay(card: Card): string {
  if (card.isBigJoker) return '大王';
  if (card.isSmallJoker) return '小王';
  
  const rankStr = getRankDisplay(card.rank as Rank);
  return `${card.suit}${rankStr}`;
}

// 获取点数显示
export function getRankDisplay(rank: Rank): string {
  const map: Record<number, string> = {
    2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '10', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'
  };
  return map[rank] || String(rank);
}

// 花色是否为红色
export function isRedSuit(suit: Suit): boolean {
  return suit === Suit.HEARTS || suit === Suit.DIAMONDS || suit === Suit.JOKER_RED;
}
