import { Card, Suit, Rank, CardType, CARD_TYPE_ORDER, RANK_JOKER_BIG } from './types';

// 创建一副牌
function createSingleDeck(): Card[] {
  const deck: Card[] = [];
  let id = 0;

  // 普通牌：2最小(值=2)，A最大(值=14)
  const suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS];
  const ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];

  for (const suit of suits) {
    for (const rank of ranks) {
      deck.push({
        id: `card-${id++}`,
        suit,
        rank: rank as Rank,
        isJoker: false,
        isBigJoker: false,
        isSmallJoker: false
      });
    }
  }

  // 小王
  deck.push({
    id: `card-${id++}`,
    suit: Suit.JOKER_BLACK,
    rank: Rank.JOKER,
    isJoker: true,
    isBigJoker: false,
    isSmallJoker: true
  });

  // 大王
  deck.push({
    id: `card-${id++}`,
    suit: Suit.JOKER_RED,
    rank: RANK_JOKER_BIG,
    isJoker: true,
    isBigJoker: true,
    isSmallJoker: false
  });

  return deck;
}

// 创建三副牌 (162张)
export function createDeck(): Card[] {
  const deck: Card[] = [];
  
  // 创建3副牌
  for (let d = 0; d < 3; d++) {
    const singleDeck = createSingleDeck();
    // 给每张牌添加副本编号以区分
    singleDeck.forEach(card => {
      card.id = `deck${d}-${card.id}`;
    });
    deck.push(...singleDeck);
  }
  
  return deck;
}

// 洗牌 (Fisher-Yates算法)
export function shuffleDeck(deck: Card[]): Card[] {
  const shuffled = [...deck];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

// 发牌给玩家
export function dealCards(deck: Card[], playerCount: number): Card[][] {
  const hands: Card[][] = Array.from({ length: playerCount }, () => []);
  const cardsPerPlayer = Math.floor(deck.length / playerCount);
  
  for (let i = 0; i < cardsPerPlayer * playerCount; i++) {
    hands[i % playerCount].push(deck[i]);
  }
  
  return hands;
}

// 获取牌的排序值 (考虑将牌)
// 排序顺序：大王 > 小王 > 将牌 > A > K > ... > 2
export function getCardSortValue(card: Card, trumpRank?: number | null): number {
  if (card.isBigJoker) return 100; // 大王最大
  if (card.isSmallJoker) return 99; // 小王第二大
  
  const rank = card.rank as number;
  
  // 如果是将牌，排在大小王后面
  if (trumpRank !== null && trumpRank !== undefined && rank === trumpRank) {
    return 98; // 将牌排在小王后面
  }
  
  return rank;
}

// 牌的排序函数 (从大到小，左大右小)
// trumpRank: 将牌点数，将牌排在大小王后面
export function sortCards(cards: Card[], _jokers: Card[] = [], trumpRank?: number | null): Card[] {
  return [...cards].sort((a, b) => getCardSortValue(b, trumpRank) - getCardSortValue(a, trumpRank));
}

// 检查是否是顺子 (杂顺)
// 大怪路子规则：顺子是5张连续的牌，不包括2
// 顺子从3开始：34567, 45678, ..., 10JQKA (A最大)
// A2345不是顺子，因为2不能参与顺子
// 但如果2是将牌，2可以在顺子里按原点数使用
// 大小王可以作为万能牌，填补顺子中的空缺
function isStraight(cards: Card[], _jokers: Card[], trumpRank?: number | null): boolean {
  if (cards.length !== 5) return false;

  // 分离大小王和普通牌
  const jokerCount = cards.filter(c => c.isJoker).length;
  const normalCards = cards.filter(c => !c.isJoker);

  // 不能有2（2不参与顺子），但如果2是将牌，可以按原点数使用
  // 例如：打2时，A2345是有效顺子
  if (normalCards.some(c => c.rank === 2 && c.rank !== trumpRank)) return false;

  // 如果没有王，检查普通顺子
  if (jokerCount === 0) {
    const sorted = sortCards(normalCards, _jokers, null);
    const values = sorted.map(c => c.rank as number);

    // 检查是否有重复
    const uniqueValues = new Set(values);
    if (uniqueValues.size !== 5) return false;

    // 检查是否是连续 (3-14, 即3-A)
    const max = Math.max(...values);
    const min = Math.min(...values);

    return max - min === 4;
  }

  // 有王的情况：检查普通牌是否可以组成顺子（允许有间隔，用王填充）
  if (normalCards.length === 0) return false; // 全是王不算顺子

  const sorted = sortCards(normalCards, _jokers, null);
  const values = sorted.map(c => c.rank as number);

  // 检查普通牌是否有重复
  const uniqueValues = new Set(values);
  if (uniqueValues.size !== values.length) return false;

  // 计算普通牌的最大最小值
  const max = Math.max(...values);
  const min = Math.min(...values);

  // 王可以填补间隔，所以：
  // 普通牌的跨度 <= 4 (因为总共5张牌)
  // 且 普通牌数量 + 王数量 = 5
  // 且 王能填补的间隔数 >= 实际间隔数
  if (max - min > 4) return false; // 跨度太大，王也填不满

  // 计算实际需要的间隔数
  let gaps = 0;
  for (let i = 1; i < values.length; i++) {
    gaps += values[i] - values[i - 1] - 1;
  }

  // 王的数量要能填补间隔
  return jokerCount >= gaps;
}

// 检查是否是同花（大小王可以作为万能牌，充当任意花色）
function isFlush(cards: Card[]): boolean {
  if (cards.length !== 5) return false;
  
  // 分离大小王和普通牌
  const normalCards = cards.filter(c => !c.isJoker);
  const jokerCount = cards.length - normalCards.length;
  
  // 如果全部是王，不算同花
  if (normalCards.length === 0) return false;
  
  // 检查普通牌是否都是同一花色
  const suits = normalCards.map(c => c.suit);
  const firstSuit = suits[0];
  
  // 所有普通牌必须是同一花色
  if (!suits.every(s => s === firstSuit)) return false;
  
  // 只要有4张或更多同花色的普通牌，加上王就可以组成同花
  // 例如：4张梅花 + 1张王 = 同花
  return normalCards.length >= 4;
}

// 检查是否是同花顺（需要同时满足同花和顺子）
// 大小王可以作为万能牌
function isStraightFlush(cards: Card[], jokers: Card[]): boolean {
  if (cards.length !== 5) return false;
  
  // 分离大小王和普通牌
  const normalCards = cards.filter(c => !c.isJoker);
  const jokerCount = cards.length - normalCards.length;
  
  // 如果全部是王，不算同花顺
  if (normalCards.length === 0) return false;
  
  // 检查普通牌是否都是同一花色
  const suits = normalCards.map(c => c.suit);
  if (!suits.every(s => s === suits[0])) return false;
  
  // 检查顺子（不含王的部分需要连续）
  if (jokerCount === 0) {
    // 没有王，必须是普通顺子
    return isStraight(cards, jokers);
  }
  
  // 有王的情况：检查普通牌是否可以组成顺子（允许有间隔，用王填充）
  const values = normalCards.map(c => c.rank as number).sort((a, b) => a - b);
  
  // 不能有2（2不参与顺子）
  if (values.includes(2)) return false;
  
  // 检查是否有重复
  const uniqueValues = new Set(values);
  if (uniqueValues.size !== values.length) return false;
  
  // 计算需要的王数量来填补间隔
  let gaps = 0;
  for (let i = 1; i < values.length; i++) {
    gaps += values[i] - values[i - 1] - 1;
  }
  
  // 王的数量要能填补间隔，且总长度为5
  // 实际牌数 + 王 = 5，且 gaps <= jokerCount
  return normalCards.length + jokerCount === 5 && gaps <= jokerCount;
}

// 识别牌型（考虑大小王作为自由牌，考虑将牌在顺子中的特殊处理）
export function recognizeCardType(cards: Card[], jokers: Card[] = [], trumpRank?: number | null): CardType | null {
  if (cards.length === 0) return null;

  // 分离大小王和普通牌
  const bigJokers = cards.filter(c => c.isBigJoker);
  const smallJokers = cards.filter(c => c.isSmallJoker);
  const normalCards = cards.filter(c => !c.isJoker);
  const totalJokers = bigJokers.length + smallJokers.length;

  if (cards.length === 1) {
    return CardType.SINGLE;
  }

  if (cards.length === 2) {
    // 2张牌：两个大王、两个小王、两个普通对子
    if (bigJokers.length === 2) {
      return CardType.PAIR; // 大王对
    }
    if (smallJokers.length === 2) {
      return CardType.PAIR; // 小王对
    }
    if (totalJokers === 1 && normalCards.length === 1) {
      // 大王+普通牌 或 小王+普通牌 视为对子（以普通牌点数为准）
      return CardType.PAIR;
    }
    // 两个普通对子
    if (normalCards.length === 2) {
      const rank1 = String(normalCards[0].rank);
      const rank2 = String(normalCards[1].rank);
      if (rank1 === rank2) {
        return CardType.PAIR;
      }
    }
    return null;
  }

  if (cards.length === 3) {
    // 3张牌：尝试用大小王组成三条
    // 2+2+小王 = 222 或 2+2+大王 = 222
    if (normalCards.length === 2) {
      const rank1 = String(normalCards[0].rank);
      const rank2 = String(normalCards[1].rank);
      if (rank1 === rank2 && totalJokers >= 1) {
        return CardType.TRIPLE;
      }
    }
    // 1+1+大小王 = 333
    if (normalCards.length === 1 && totalJokers >= 2) {
      return CardType.TRIPLE;
    }
    // 3张相同
    const rank1 = cards[0].isJoker ? 'joker' : String(cards[0].rank);
    const rank2 = cards[1].isJoker ? 'joker' : String(cards[1].rank);
    const rank3 = cards[2].isJoker ? 'joker' : String(cards[2].rank);
    if (rank1 === rank2 && rank2 === rank3) {
      return CardType.TRIPLE;
    }
    return null;
  }

  if (cards.length === 5) {
    // 5张牌：优先用大小王组成最大牌型
    // 先检查五同（最重要！）
    const fiveResult = checkFiveOfKindWithJokers(normalCards, bigJokers.length, smallJokers.length);
    if (fiveResult.isFiveOfKind) {
      return CardType.FIVE_OF_KIND;
    }

    // 检查同花顺
    if (isStraightFlush(cards, jokers)) return CardType.STRAIGHT_FLUSH;

    // 检查四带一
    const fourResult = checkFourWithOneWithJokers(normalCards, bigJokers.length, smallJokers.length);
    if (fourResult.isFourWithOne) {
      return CardType.FOUR_WITH_ONE;
    }

    // 检查三带二
    const threeResult = checkThreeWithTwoWithJokers(normalCards, bigJokers.length, smallJokers.length);
    if (threeResult.isThreeWithTwo) {
      return CardType.THREE_WITH_TWO;
    }

    // 检查同花（不能有大小王）
    if (isFlush(cards)) return CardType.FLUSH;

    // 检查顺子
    if (isStraight(cards, jokers, trumpRank)) return CardType.STRAIGHT;
  }

  return null;
}

// 检查是否能用大小王组成五同
function checkFiveOfKindWithJokers(normalCards: Card[], bigJokers: number, smallJokers: number): { isFiveOfKind: boolean, mainRank: number } {
  // 统计普通牌的点数
  const rankCounts: Record<number, number> = {};
  for (const c of normalCards) {
    rankCounts[c.rank as number] = (rankCounts[c.rank as number] || 0) + 1;
  }

  const jokerCount = bigJokers + smallJokers;

  // 检查是否可以通过添加大小王组成五同
  for (const [rank, count] of Object.entries(rankCounts)) {
    const total = count + jokerCount;
    if (total >= 5) {
      return { isFiveOfKind: true, mainRank: Number(rank) };
    }
  }

  // 检查是否可以用多个不同点数+大小王组成（不可能，因为需要5张相同）
  return { isFiveOfKind: false, mainRank: 0 };
}

// 大小王自由牌规则：
// 1. 当大小王可以组成不同牌型时，优先组成最大的牌型
//    例如：3334+王 → 视作33334（四带一），而不是33344（三带二）
// 2. 在同一牌型中，王补充到最小点数位置
//    例如：大王+3344 → 33344（王充当3），而不是33444
//         小王+3456 → 23456（王充当2），而不是34567
// 3. 多张王的比较规则：
//    3大王 > 2大王1小王 = 1大王2小王 = 3小王
//    2大王 > 大王小王 = 2小王

// 检查是否能用大小王组成四带一
// 关键规则：王优先补充到最小点数，形成四条
function checkFourWithOneWithJokers(normalCards: Card[], bigJokers: number, smallJokers: number): { isFourWithOne: boolean, mainRank: number } {
  const rankCounts: Record<number, number> = {};
  for (const c of normalCards) {
    rankCounts[c.rank as number] = (rankCounts[c.rank as number] || 0) + 1;
  }

  const jokerCount = bigJokers + smallJokers;
  const ranks = Object.keys(rankCounts).map(Number).sort((a, b) => a - b); // 从小到大排序

  // 检查是否能组成四带一
  // 规则：王补充到最小点数优先
  for (const rank of ranks) {
    const count = rankCounts[rank];
    if (count + jokerCount >= 4) {
      // 检查是否还有其他牌作为单牌（或王作为单牌）
      const otherCards = ranks.filter(r => r !== rank);
      const totalOther = otherCards.reduce((sum, r) => sum + rankCounts[r], 0);
      if (totalOther >= 1 || jokerCount >= (4 - count) + 1) {
        return { isFourWithOne: true, mainRank: rank };
      }
    }
  }

  return { isFourWithOne: false, mainRank: 0 };
}

// 检查是否能用大小王组成三带二
// 规则：王补充到最小点数优先
function checkThreeWithTwoWithJokers(normalCards: Card[], bigJokers: number, smallJokers: number): { isThreeWithTwo: boolean, mainRank: number } {
  const rankCounts: Record<number, number> = {};
  for (const c of normalCards) {
    rankCounts[c.rank as number] = (rankCounts[c.rank as number] || 0) + 1;
  }

  const jokerCount = bigJokers + smallJokers;
  const ranks = Object.keys(rankCounts).map(Number).sort((a, b) => a - b);

  // 规则：王优先补充到最小点数
  // 首先检查是否能组成三带二（不使用王作为单牌）
  for (const threeRank of ranks) {
    const threeCount = rankCounts[threeRank];
    if (threeCount + jokerCount >= 3) {
      // 找另一对（可以是现有的对子，也可以是单牌+王组成的对子）
      const remainingJokers = jokerCount - Math.max(0, 3 - threeCount);
      for (const twoRank of ranks) {
        if (twoRank === threeRank) continue;
        const twoCount = rankCounts[twoRank];
        // 现有对子，或单牌+剩余王组成对子
        if (twoCount >= 2 || (twoCount >= 1 && remainingJokers >= 1)) {
          return { isThreeWithTwo: true, mainRank: threeRank };
        }
      }
    }
  }

  return { isThreeWithTwo: false, mainRank: 0 };
}

// 计算王的"质量值"（用于比较多张王的牌型）
// 规则：3大王 > 2大王1小王 = 1大王2小王 = 3小王
//      2大王 > 大王小王 = 2小王
// 返回值越大，牌型越大
function getJokerQuality(bigJokerCount: number, smallJokerCount: number): number {
  const total = bigJokerCount + smallJokerCount;
  // 基础值：总张数 * 100
  // 大王加权：每个大王 +10
  // 这样：300+30=330(3大) > 300+20=320(2大1小) = 300+10=310(1大2小) = 300+0=300(3小)
  return total * 100 + bigJokerCount * 10;
}

// 获取牌型的主要比较值（考虑大小王作为自由牌和将牌）
// trumpRank: 将牌点数（如打A时=14，打2时=2）
// 关键规则：王补充到最小点数位置
function getCardTypeMainValue(cards: Card[], trumpRank?: number | null): number {
  if (cards.length === 0) return 0;

  // 分离大小王和普通牌
  const bigJokers = cards.filter(c => c.isBigJoker);
  const smallJokers = cards.filter(c => c.isSmallJoker);
  const normalCards = cards.filter(c => !c.isJoker);
  const bigJokerCount = bigJokers.length;
  const smallJokerCount = smallJokers.length;
  const totalJokers = bigJokerCount + smallJokerCount;

  // 统计普通牌的点数
  const rankCounts: Record<number, number> = {};
  for (const c of normalCards) {
    rankCounts[c.rank as number] = (rankCounts[c.rank as number] || 0) + 1;
  }

  const type = recognizeCardType(cards, [], trumpRank);

  // 五同：比较组成五同的点数
  // 关键规则：王补充到最小点数
  if (type === CardType.FIVE_OF_KIND) {
    // 如果全是王，按王的组成比较
    if (normalCards.length === 0) {
      // 返回王的质量值 + 一个高基础值
      return 1000 + getJokerQuality(bigJokerCount, smallJokerCount);
    }
    
    // 有普通牌的情况：王补充到该点数，形成五同
    // 找到最小点数（王补充到最小点数）
    const ranks = Object.keys(rankCounts).map(Number).sort((a, b) => a - b);
    const mainRank = ranks[0]; // 王补充到最小点数
    
    // 如果这个五同的点数是将牌，加上偏移
    if (trumpRank !== null && mainRank === trumpRank) {
      return 15.5; // 将牌五同，仅次于王炸
    }
    return mainRank;
  }

  // 四带一：比较四条的点
  // 关键规则：王补充到最小点数形成四条
  if (type === CardType.FOUR_WITH_ONE) {
    // 找到最小点数（王补充到最小点数形成四条）
    const ranks = Object.keys(rankCounts).map(Number).sort((a, b) => a - b);
    const mainRank = ranks[0];
    
    // 如果这个四条的点数是将牌，加上偏移
    if (trumpRank !== null && mainRank === trumpRank) {
      return 15.5;
    }
    return mainRank;
  }

  // 三带二：比较三张的点
  // 关键规则：王补充到最小点数形成三条
  if (type === CardType.THREE_WITH_TWO) {
    // 找到最小点数（王补充到最小点数形成三条）
    const ranks = Object.keys(rankCounts).map(Number).sort((a, b) => a - b);
    const mainRank = ranks[0];
    
    // 如果这个三张的点数是将牌，加上偏移
    if (trumpRank !== null && mainRank === trumpRank) {
      return 15.5;
    }
    return mainRank;
  }

  // 对子：特殊处理
  // 规则：2大王 > 大王小王 = 2小王
  // 大小王的值：大王=17，小王=16
  if (type === CardType.PAIR) {
    // 全是王的情况
    if (normalCards.length === 0) {
      // 两个大王：最大
      if (bigJokerCount === 2) return 200;
      // 大王小王 或 两个小王：相等
      if (bigJokerCount === 1 && smallJokerCount === 1) return 160;
      if (smallJokerCount === 2) return 160;
      return 0;
    }
    // 王+普通牌：王补充到该点数
    if (totalJokers === 1 && normalCards.length === 1) {
      const pairRank = normalCards[0].rank as number;
      if (trumpRank !== null && pairRank === trumpRank) {
        return 15.5; // 将牌对子
      }
      return pairRank;
    }
    // 普通对子
    if (normalCards.length === 2) {
      const pairRank = normalCards[0].rank as number;
      if (trumpRank !== null && pairRank === trumpRank) {
        return 15.5;
      }
      return pairRank;
    }
    return 0;
  }

  // 三条：特殊处理（需要考虑将牌和多张王的比较）
  // 规则：3大王 > 2大王1小王 = 1大王2小王 = 3小王
  if (type === CardType.TRIPLE) {
    // 全是王的情况
    if (normalCards.length === 0) {
      return 300 + getJokerQuality(bigJokerCount, smallJokerCount);
    }
    
    // 有普通牌：王补充到最小点数
    let tripleRank = 0;
    if (normalCards.length >= 2) {
      tripleRank = normalCards[0].rank as number;
    } else if (normalCards.length === 1 && totalJokers >= 2) {
      tripleRank = normalCards[0].rank as number;
    } else if (normalCards.length === 3) {
      tripleRank = normalCards[0].rank as number;
    }
    
    if (trumpRank !== null && tripleRank === trumpRank) {
      return 15.5;
    }
    return tripleRank;
  }

  // 单张牌：特殊处理大小王
  if (type === CardType.SINGLE) {
    // 大王
    if (bigJokerCount === 1) {
      return 17; // 大王最大
    }
    // 小王
    if (smallJokerCount === 1) {
      return 16; // 小王第二大
    }
    // 普通单张
    if (normalCards.length === 1) {
      const cardRank = normalCards[0].rank as number;
      // 如果是将牌，加上偏移
      if (trumpRank !== null && cardRank === trumpRank) {
        return 15.5; // 将牌，仅次于大小王
      }
      return cardRank;
    }
    return 0;
  }

  // 同花：比较最大牌的点数，将花比普通同花大
  // 注意：同花中有王时，用除王以外最大的牌来计算
  if (type === CardType.FLUSH) {
    // 找出最大牌的点数（不含王）
    const maxRank = normalCards.length > 0 
      ? Math.max(...normalCards.map(c => c.rank as number))
      : 0;
    
    // 检查是否包含将牌点数（将花）
    const hasTrump = trumpRank !== null && normalCards.some(c => c.rank === trumpRank);
    
    // 如果是将花（包含将牌点数），值设为15.5（仅次于大小王）
    if (hasTrump) {
      return 15.5;
    }
    
    return maxRank;
  }

  // 顺子：比较最大牌的点数，将牌按原点数算，没有"将顺"的说法
  if (type === CardType.STRAIGHT) {
    const maxRank = normalCards.length > 0 
      ? Math.max(...normalCards.map(c => c.rank as number))
      : 0;
    return maxRank;
  }

  // 其他牌型：找出出现次数最多的点数
  let maxCount = 0;
  let mainRank = 0;
  for (const [rank, count] of Object.entries(rankCounts)) {
    const total = count + totalJokers;
    if (total > maxCount || (total === maxCount && Number(rank) > mainRank)) {
      maxCount = total;
      mainRank = Number(rank);
    }
  }

  // 如果有将牌，将将牌的点数加上偏移，使其比普通牌大但比大小王小
  if (trumpRank !== null) {
    // 如果主要点数是将牌点数，加上偏移
    if (mainRank === trumpRank) {
      mainRank = 15.5; // 将牌，仅次于大小王
    }
    // 如果没有普通牌，但有将牌，使用将牌点数
    if (mainRank === 0 && trumpRank && trumpRank > 0) {
      mainRank = 15.5;
    }
  }

  return mainRank;
}

// 比较两副牌的大小
// trumpRank: 将牌点数（如打A时=14，打2时=2），将牌比其他普通牌都大，但比大小王小
export function compareCards(cards1: Card[], cards2: Card[], jokers: Card[], trumpRank?: number | null): number {
  const type1 = recognizeCardType(cards1, jokers, trumpRank);
  const type2 = recognizeCardType(cards2, jokers, trumpRank);

  if (!type1 || !type2) return 0;

  // 首先比较牌型
  const order1 = CARD_TYPE_ORDER[type1];
  const order2 = CARD_TYPE_ORDER[type2];

  if (order1 > order2) return 1;
  if (order1 < order2) return -1;

  // 牌型相同时，比较主要点数
  const mainValue1 = getCardTypeMainValue(cards1, trumpRank);
  const mainValue2 = getCardTypeMainValue(cards2, trumpRank);

  if (mainValue1 > mainValue2) return 1;
  if (mainValue1 < mainValue2) return -1;

  return 0;
}

// AI决策 - 选择最佳出牌
// currentRoundCardCount: 如果指定了数量，AI必须出这个数量的牌
// trumpRank: 将牌点数
export function aiSelectCards(hand: Card[], jokers: Card[], currentPlayed: Card[] | null, isFirstPlay: boolean, requiredCardCount?: number, trumpRank?: number | null): Card[] | null {
  const cardType = recognizeCardType(currentPlayed || [], jokers, trumpRank);

  // 首发牌：找最小的有效牌型
  if (isFirstPlay || !currentPlayed || currentPlayed.length === 0) {
    const sorted = sortCards(hand, jokers, trumpRank);

    // 如果指定了牌数，找出该数量的最小牌型
    if (requiredCardCount) {
      if (requiredCardCount === 1) {
        // 出最小的单张
        return sorted.length > 0 ? [sorted[sorted.length - 1]] : null;
      } else if (requiredCardCount === 2) {
        // 找最小的对子
        for (let i = sorted.length - 1; i >= 1; i--) {
          if (sorted[i].rank === sorted[i - 1].rank && !sorted[i].isJoker && !sorted[i - 1].isJoker) {
            return [sorted[i], sorted[i - 1]];
          }
        }
        // 没有对子，尝试用王+最小牌组成对子
        const jokerCards = sorted.filter(c => c.isJoker);
        const normalCards = sorted.filter(c => !c.isJoker);
        if (jokerCards.length >= 1 && normalCards.length >= 1) {
          return [jokerCards[0], normalCards[normalCards.length - 1]];
        }
        return null;
      } else if (requiredCardCount === 3) {
        // 找最小的三条
        for (let i = sorted.length - 1; i >= 2; i--) {
          if (sorted[i].rank === sorted[i - 1].rank && sorted[i - 1].rank === sorted[i - 2].rank && !sorted[i].isJoker) {
            return [sorted[i], sorted[i - 1], sorted[i - 2]];
          }
        }
        // 没有三条，尝试用王凑
        const jokerCards = sorted.filter(c => c.isJoker);
        const normalCards = sorted.filter(c => !c.isJoker);
        // 2张相同+1王 = 三条
        for (let i = normalCards.length - 1; i >= 1; i--) {
          if (normalCards[i].rank === normalCards[i - 1].rank) {
            if (jokerCards.length >= 1) {
              return [normalCards[i], normalCards[i - 1], jokerCards[0]];
            }
          }
        }
        return null;
      } else if (requiredCardCount === 5) {
        // 找最小的5张牌型
        const fiveCombo = findSmallestFiveCardCombo(hand, jokers, trumpRank);
        return fiveCombo;
      }
    }

    // 没有指定牌数，出最小的单张
    if (sorted.length > 0) {
      return [sorted[sorted.length - 1]];
    }
    return null;
  }

  // 跟牌：找能压过的最小牌型
  if (cardType && currentPlayed.length > 0) {
    const currentType = cardType;

    // 如果指定了牌数，必须出这个数量
    const targetCount = requiredCardCount || currentPlayed.length;

    // 尝试找能压过的牌（从最小的开始找）
    const sortedHand = sortCards(hand, jokers, trumpRank).reverse(); // 从小到大排序

    for (let i = 0; i < sortedHand.length; i++) {
      if (targetCount === 1) {
        // 单张：使用比较函数判断能否压过
        if (compareCards([sortedHand[i]], currentPlayed, jokers, trumpRank) > 0) {
          return [sortedHand[i]];
        }
      } else if (targetCount === 2 && currentType === CardType.PAIR) {
        // 对子
        if (i + 1 < sortedHand.length && sortedHand[i].rank === sortedHand[i + 1].rank) {
          const pair = [sortedHand[i], sortedHand[i + 1]];
          if (compareCards(pair, currentPlayed, jokers, trumpRank) > 0) {
            return pair;
          }
        }
      } else if (targetCount === 3 && currentType === CardType.TRIPLE) {
        // 三张
        if (i + 2 < sortedHand.length &&
            sortedHand[i].rank === sortedHand[i + 1].rank &&
            sortedHand[i + 1].rank === sortedHand[i + 2].rank) {
          const triple = [sortedHand[i], sortedHand[i + 1], sortedHand[i + 2]];
          if (compareCards(triple, currentPlayed, jokers, trumpRank) > 0) {
            return triple;
          }
        }
      } else if (targetCount === 5) {
        // 五张牌型：顺子、同花、三带二、四带一、五同、同花顺
        // 需要遍历所有可能的5张组合，找能压过的最小组合
        const fiveCardCombos = findFiveCardCombos(hand, jokers, currentPlayed, trumpRank);
        if (fiveCardCombos && fiveCardCombos.length > 0) {
          // 返回能压过的最小的组合
          return fiveCardCombos;
        }
        // 如果找不到能压过的组合，直接跳出循环，选择不出
        break;
      }
    }
  }

  // 找不到合适的牌，选择不出
  return null;
}

// 找最小的5张牌型组合（用于首发）
function findSmallestFiveCardCombo(hand: Card[], jokers: Card[], trumpRank?: number | null): Card[] | null {
  if (hand.length < 5) return null;

  const jokerCards = hand.filter(c => c.isJoker);
  const normalCards = hand.filter(c => !c.isJoker);

  // 统计每张牌的数量
  const rankCounts: Record<number, Card[]> = {};
  for (const card of normalCards) {
    const rank = card.rank as number;
    if (!rankCounts[rank]) rankCounts[rank] = [];
    rankCounts[rank].push(card);
  }

  const ranks = Object.keys(rankCounts).map(Number).sort((a, b) => a - b); // 从小到大

  // 按牌型优先级找最小的组合：
  // 顺子 < 同花 < 三带二 < 四带一 < 五同（首发时优先出小牌型）

  // 1. 找最小的顺子
  const straightCombo = findSmallestStraight(normalCards, jokers, trumpRank);
  if (straightCombo) return straightCombo;

  // 2. 找最小的同花
  const flushCombo = findSmallestFlush(normalCards, jokerCards, trumpRank);
  if (flushCombo) return flushCombo;

  // 3. 找最小的三带二
  const threeWithTwo = findSmallestThreeWithTwo(normalCards, jokerCards, rankCounts, ranks, trumpRank);
  if (threeWithTwo) return threeWithTwo;

  // 4. 找最小的四带一
  const fourWithOne = findSmallestFourWithOne(normalCards, jokerCards, rankCounts, ranks, trumpRank);
  if (fourWithOne) return fourWithOne;

  // 5. 找最小的五同
  const fiveOfKind = findSmallestFiveOfKind(normalCards, jokerCards, rankCounts, ranks, trumpRank);
  if (fiveOfKind) return fiveOfKind;

  return null;
}

// 找最小的顺子
function findSmallestStraight(normalCards: Card[], _jokers: Card[], _trumpRank?: number | null): Card[] | null {
  // 排除2和王
  const cards = normalCards.filter(c => c.rank !== 2);
  if (cards.length < 5) return null;

  // 按点数分组
  const rankCards: Record<number, Card> = {};
  for (const card of cards) {
    const rank = card.rank as number;
    if (!rankCards[rank]) {
      rankCards[rank] = card;
    }
  }

  const sortedRanks = Object.keys(rankCards).map(Number).sort((a, b) => a - b);

  // 找最小的连续5个点数
  for (let i = 0; i <= sortedRanks.length - 5; i++) {
    let isConsecutive = true;
    for (let j = 0; j < 4; j++) {
      if (sortedRanks[i + j + 1] - sortedRanks[i + j] !== 1) {
        isConsecutive = false;
        break;
      }
    }
    if (isConsecutive) {
      const combo: Card[] = [];
      for (let j = 0; j < 5; j++) {
        combo.push(rankCards[sortedRanks[i + j]]);
      }
      return combo;
    }
  }

  return null;
}

// 找最小的同花
function findSmallestFlush(normalCards: Card[], jokerCards: Card[], _trumpRank?: number | null): Card[] | null {
  // 按花色分组
  const suitCards: Record<string, Card[]> = {};
  for (const card of normalCards) {
    const suit = card.suit;
    if (!suitCards[suit]) suitCards[suit] = [];
    suitCards[suit].push(card);
  }

  // 找每张花色中最小的同花组合
  let smallestFlush: Card[] | null = null;
  let smallestValue = Infinity;

  for (const suit of Object.keys(suitCards)) {
    const cards = suitCards[suit];
    // 需要至少4张同花色才能加王组成同花
    if (cards.length >= 4) {
      const sorted = cards.sort((a, b) => (a.rank as number) - (b.rank as number)); // 从小到大

      if (cards.length >= 5) {
        // 取最小的5张
        const combo = sorted.slice(0, 5);
        const value = Math.max(...combo.map(c => c.rank as number));
        if (value < smallestValue) {
          smallestValue = value;
          smallestFlush = combo;
        }
      }
      // 4张同花色 + 1张王
      if (cards.length >= 4 && jokerCards.length >= 1) {
        const combo = [...sorted.slice(0, 4), jokerCards[0]];
        const value = Math.max(...sorted.slice(0, 4).map(c => c.rank as number));
        if (value < smallestValue) {
          smallestValue = value;
          smallestFlush = combo;
        }
      }
    }
  }

  return smallestFlush;
}

// 找最小的三带二
function findSmallestThreeWithTwo(normalCards: Card[], jokerCards: Card[], rankCounts: Record<number, Card[]>, ranks: number[], _trumpRank?: number | null): Card[] | null {
  // 找最小的三条
  for (const rank of ranks) {
    const threeCards = rankCounts[rank];
    if (threeCards.length >= 3) {
      // 找另一对
      for (const otherRank of ranks) {
        if (otherRank === rank) continue;
        const twoCards = rankCounts[otherRank];
        if (twoCards.length >= 2) {
          return [...threeCards.slice(0, 3), ...twoCards.slice(0, 2)];
        }
      }
      // 找单牌+王凑对子
      if (jokerCards.length >= 1) {
        for (const otherRank of ranks) {
          if (otherRank === rank) continue;
          const oneCard = rankCounts[otherRank][0];
          return [...threeCards.slice(0, 3), oneCard, jokerCards[0]];
        }
      }
    }
    // 2张+1王=三条，找另一对
    if (threeCards.length >= 2 && jokerCards.length >= 1) {
      for (const otherRank of ranks) {
        if (otherRank === rank) continue;
        const twoCards = rankCounts[otherRank];
        if (twoCards.length >= 2) {
          return [...threeCards.slice(0, 2), jokerCards[0], ...twoCards.slice(0, 2)];
        }
      }
    }
  }

  return null;
}

// 找最小的四带一
function findSmallestFourWithOne(normalCards: Card[], jokerCards: Card[], rankCounts: Record<number, Card[]>, ranks: number[], _trumpRank?: number | null): Card[] | null {
  // 找最小的四条
  for (const rank of ranks) {
    const fourCards = rankCounts[rank];
    if (fourCards.length >= 4) {
      // 找一张单牌
      for (const otherRank of ranks) {
        if (otherRank === rank) continue;
        return [...fourCards.slice(0, 4), rankCounts[otherRank][0]];
      }
      // 用王作为单牌
      if (jokerCards.length >= 1) {
        return [...fourCards.slice(0, 4), jokerCards[0]];
      }
    }
    // 3张+1王=四条
    if (fourCards.length >= 3 && jokerCards.length >= 1) {
      for (const otherRank of ranks) {
        if (otherRank === rank) continue;
        return [...fourCards.slice(0, 3), jokerCards[0], rankCounts[otherRank][0]];
      }
    }
  }

  return null;
}

// 找最小的五同
function findSmallestFiveOfKind(normalCards: Card[], jokerCards: Card[], rankCounts: Record<number, Card[]>, ranks: number[], _trumpRank?: number | null): Card[] | null {
  for (const rank of ranks) {
    const cards = rankCounts[rank];
    const needed = 5 - cards.length;
    if (jokerCards.length >= needed) {
      return [...cards, ...jokerCards.slice(0, needed)];
    }
    if (cards.length >= 5) {
      return cards.slice(0, 5);
    }
  }

  return null;
}

// 计算joker的数量
export function countJokers(cards: Card[]): number {
  return cards.filter(c => c.isJoker).length;
}

// 找出所有能压过当前牌的5张牌组合
function findFiveCardCombos(hand: Card[], jokers: Card[], currentPlayed: Card[], trumpRank?: number | null): Card[] | null {
  if (hand.length < 5) return null;

  const currentType = recognizeCardType(currentPlayed, jokers, trumpRank);
  if (!currentType) return null;

  // 存储所有能压过的组合
  const validCombos: { cards: Card[], value: number, typeOrder: number }[] = [];

  // 统计手牌中每张牌的数量
  const rankCounts: Record<number, Card[]> = {};
  const jokerCards: Card[] = [];

  for (const card of hand) {
    if (card.isJoker) {
      jokerCards.push(card);
    } else {
      const rank = card.rank as number;
      if (!rankCounts[rank]) rankCounts[rank] = [];
      rankCounts[rank].push(card);
    }
  }

  const ranks = Object.keys(rankCounts).map(Number);

  // 1. 检查五同（五张相同）
  for (const rank of ranks) {
    const cards = rankCounts[rank];
    const needed = 5 - cards.length;
    if (jokerCards.length >= needed) {
      const combo = [...cards, ...jokerCards.slice(0, needed)];
      if (combo.length === 5 && recognizeCardType(combo, jokers, trumpRank) === CardType.FIVE_OF_KIND) {
        const typeOrder = CARD_TYPE_ORDER[CardType.FIVE_OF_KIND];
        const mainValue = getCardTypeMainValue(combo, trumpRank);
        if (compareCards(combo, currentPlayed, jokers, trumpRank) > 0) {
          validCombos.push({ cards: combo, value: mainValue, typeOrder });
        }
      }
    }
    // 5张同点数不需要王
    if (cards.length >= 5) {
      const combo = cards.slice(0, 5);
      if (recognizeCardType(combo, jokers, trumpRank) === CardType.FIVE_OF_KIND) {
        const typeOrder = CARD_TYPE_ORDER[CardType.FIVE_OF_KIND];
        const mainValue = getCardTypeMainValue(combo, trumpRank);
        if (compareCards(combo, currentPlayed, jokers, trumpRank) > 0) {
          validCombos.push({ cards: combo, value: mainValue, typeOrder });
        }
      }
    }
  }

  // 2. 检查同花顺
  const flushStraightCombos = findFlushStraightCombos(hand, jokers, trumpRank);
  for (const combo of flushStraightCombos) {
    if (compareCards(combo, currentPlayed, jokers, trumpRank) > 0) {
      const typeOrder = CARD_TYPE_ORDER[CardType.STRAIGHT_FLUSH];
      const mainValue = getCardTypeMainValue(combo, trumpRank);
      validCombos.push({ cards: combo, value: mainValue, typeOrder });
    }
  }

  // 3. 检查四带一
  for (const rank of ranks) {
    const fourCards = rankCounts[rank];
    if (fourCards.length >= 4) {
      // 找一张单牌
      for (const otherRank of ranks) {
        if (otherRank === rank) continue;
        const singleCard = rankCounts[otherRank][0];
        const combo = [...fourCards.slice(0, 4), singleCard];
        if (recognizeCardType(combo, jokers, trumpRank) === CardType.FOUR_WITH_ONE) {
          const typeOrder = CARD_TYPE_ORDER[CardType.FOUR_WITH_ONE];
          const mainValue = getCardTypeMainValue(combo, trumpRank);
          if (compareCards(combo, currentPlayed, jokers, trumpRank) > 0) {
            validCombos.push({ cards: combo, value: mainValue, typeOrder });
          }
        }
      }
      // 用王作为单牌
      if (jokerCards.length >= 1) {
        const combo = [...fourCards.slice(0, 4), jokerCards[0]];
        if (recognizeCardType(combo, jokers, trumpRank) === CardType.FOUR_WITH_ONE) {
          const typeOrder = CARD_TYPE_ORDER[CardType.FOUR_WITH_ONE];
          const mainValue = getCardTypeMainValue(combo, trumpRank);
          if (compareCards(combo, currentPlayed, jokers, trumpRank) > 0) {
            validCombos.push({ cards: combo, value: mainValue, typeOrder });
          }
        }
      }
    }
    // 3张+1张王=四带一
    if (fourCards.length >= 3 && jokerCards.length >= 1) {
      for (const otherRank of ranks) {
        if (otherRank === rank) continue;
        const singleCard = rankCounts[otherRank][0];
        const combo = [...fourCards.slice(0, 3), jokerCards[0], singleCard];
        if (recognizeCardType(combo, jokers, trumpRank) === CardType.FOUR_WITH_ONE) {
          const typeOrder = CARD_TYPE_ORDER[CardType.FOUR_WITH_ONE];
          const mainValue = getCardTypeMainValue(combo, trumpRank);
          if (compareCards(combo, currentPlayed, jokers, trumpRank) > 0) {
            validCombos.push({ cards: combo, value: mainValue, typeOrder });
          }
        }
      }
    }
  }

  // 4. 检查三带二
  for (const threeRank of ranks) {
    const threeCards = rankCounts[threeRank];
    if (threeCards.length >= 3) {
      for (const twoRank of ranks) {
        if (twoRank === threeRank) continue;
        const twoCards = rankCounts[twoRank];
        if (twoCards.length >= 2) {
          const combo = [...threeCards.slice(0, 3), ...twoCards.slice(0, 2)];
          if (recognizeCardType(combo, jokers, trumpRank) === CardType.THREE_WITH_TWO) {
            const typeOrder = CARD_TYPE_ORDER[CardType.THREE_WITH_TWO];
            const mainValue = getCardTypeMainValue(combo, trumpRank);
            if (compareCards(combo, currentPlayed, jokers, trumpRank) > 0) {
              validCombos.push({ cards: combo, value: mainValue, typeOrder });
            }
          }
        }
      }
    }
    // 2张+1张王=三张，找另一对
    if (threeCards.length >= 2 && jokerCards.length >= 1) {
      for (const twoRank of ranks) {
        if (twoRank === threeRank) continue;
        const twoCards = rankCounts[twoRank];
        if (twoCards.length >= 2) {
          const combo = [...threeCards.slice(0, 2), jokerCards[0], ...twoCards.slice(0, 2)];
          if (recognizeCardType(combo, jokers, trumpRank) === CardType.THREE_WITH_TWO) {
            const typeOrder = CARD_TYPE_ORDER[CardType.THREE_WITH_TWO];
            const mainValue = getCardTypeMainValue(combo, trumpRank);
            if (compareCards(combo, currentPlayed, jokers, trumpRank) > 0) {
              validCombos.push({ cards: combo, value: mainValue, typeOrder });
            }
          }
        }
      }
    }
  }

  // 5. 检查同花
  const flushCombos = findFlushCombos(hand, jokers, trumpRank);
  for (const combo of flushCombos) {
    if (compareCards(combo, currentPlayed, jokers, trumpRank) > 0) {
      const typeOrder = CARD_TYPE_ORDER[CardType.FLUSH];
      const mainValue = getCardTypeMainValue(combo, trumpRank);
      validCombos.push({ cards: combo, value: mainValue, typeOrder });
    }
  }

  // 6. 检查顺子
  const straightCombos = findStraightCombos(hand, jokers, trumpRank);
  for (const combo of straightCombos) {
    if (compareCards(combo, currentPlayed, jokers, trumpRank) > 0) {
      const typeOrder = CARD_TYPE_ORDER[CardType.STRAIGHT];
      const mainValue = getCardTypeMainValue(combo, trumpRank);
      validCombos.push({ cards: combo, value: mainValue, typeOrder });
    }
  }

  // 按牌型优先级和点数排序，返回最小的能压过的组合
  if (validCombos.length === 0) return null;

  validCombos.sort((a, b) => {
    // 先按牌型排序（小的优先）
    if (a.typeOrder !== b.typeOrder) return a.typeOrder - b.typeOrder;
    // 再按点数排序（小的优先）
    return a.value - b.value;
  });

  return validCombos[0].cards;
}

// 找同花组合
function findFlushCombos(hand: Card[], jokers: Card[], _trumpRank?: number | null): Card[][] {
  const results: Card[][] = [];
  const jokerCards = hand.filter(c => c.isJoker);
  const normalCards = hand.filter(c => !c.isJoker);

  // 按花色分组
  const suitCards: Record<string, Card[]> = {};
  for (const card of normalCards) {
    const suit = card.suit;
    if (!suitCards[suit]) suitCards[suit] = [];
    suitCards[suit].push(card);
  }

  // 找每张花色的同花
  for (const suit of Object.keys(suitCards)) {
    const cards = suitCards[suit];
    // 如果有4张以上同花色，可以加王组成同花
    if (cards.length >= 4) {
      // 5张同花色
      if (cards.length >= 5) {
        // 取最小的5张
        const sorted = sortCards(cards, jokers, _trumpRank).reverse(); // 从小到大
        for (let i = 0; i <= sorted.length - 5; i++) {
          results.push(sorted.slice(i, i + 5));
        }
      }
      // 4张同花色 + 1张王
      if (cards.length >= 4 && jokerCards.length >= 1) {
        const sorted = sortCards(cards, jokers, _trumpRank).reverse();
        results.push([...sorted.slice(0, 4), jokerCards[0]]);
      }
    }
  }

  return results;
}

// 找顺子组合（不含王，因为王不参与顺子）
function findStraightCombos(hand: Card[], jokers: Card[], _trumpRank?: number | null): Card[][] {
  const results: Card[][] = [];
  const normalCards = hand.filter(c => !c.isJoker);

  // 排除2
  const cardsNo2 = normalCards.filter(c => c.rank !== 2);

  // 按点数分组（可能有重复）
  const rankCards: Record<number, Card[]> = {};
  for (const card of cardsNo2) {
    const rank = card.rank as number;
    if (!rankCards[rank]) rankCards[rank] = [];
    rankCards[rank].push(card);
  }

  const sortedRanks = Object.keys(rankCards).map(Number).sort((a, b) => a - b);

  // 找连续5个点数
  for (let i = 0; i <= sortedRanks.length - 5; i++) {
    let isConsecutive = true;
    for (let j = 0; j < 4; j++) {
      if (sortedRanks[i + j + 1] - sortedRanks[i + j] !== 1) {
        isConsecutive = false;
        break;
      }
    }
    if (isConsecutive) {
      const combo: Card[] = [];
      for (let j = 0; j < 5; j++) {
        combo.push(rankCards[sortedRanks[i + j]][0]);
      }
      results.push(combo);
    }
  }

  return results;
}

// 找同花顺组合
function findFlushStraightCombos(hand: Card[], jokers: Card[], _trumpRank?: number | null): Card[][] {
  const results: Card[][] = [];
  const jokerCards = hand.filter(c => c.isJoker);
  const normalCards = hand.filter(c => !c.isJoker && c.rank !== 2); // 排除王和2

  // 按花色分组
  const suitCards: Record<string, Card[]> = {};
  for (const card of normalCards) {
    const suit = card.suit;
    if (!suitCards[suit]) suitCards[suit] = [];
    suitCards[suit].push(card);
  }

  for (const suit of Object.keys(suitCards)) {
    const cards = suitCards[suit];
    const ranks = [...new Set(cards.map(c => c.rank as number))].sort((a, b) => a - b);

    // 找连续5个点数
    for (let i = 0; i <= ranks.length - 5; i++) {
      let isConsecutive = true;
      for (let j = 0; j < 4; j++) {
        if (ranks[i + j + 1] - ranks[i + j] !== 1) {
          isConsecutive = false;
          break;
        }
      }
      if (isConsecutive) {
        const combo: Card[] = [];
        for (let j = 0; j < 5; j++) {
          const card = cards.find(c => c.rank === ranks[i + j]);
          if (card) combo.push(card);
        }
        if (combo.length === 5) {
          results.push(combo);
        }
      }
    }

    // 用王填充的同花顺（简化处理，暂时跳过复杂的王填充逻辑）
  }

  return results;
}
