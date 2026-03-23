// gameLogic.js - 大怪路子游戏逻辑（从 TypeScript 移植）

export const Suit = {
  SPADES: '♠',
  HEARTS: '♥',
  CLUBS: '♣',
  DIAMONDS: '♦',
  JOKER_RED: 'joker-red',
  JOKER_BLACK: 'joker-black'
};

export const RANK_JOKER_BIG = 17;

export const CardType = {
  SINGLE: 'single',
  PAIR: 'pair',
  TRIPLE: 'triple',
  STRAIGHT: 'straight',
  FLUSH: 'flush',
  THREE_WITH_TWO: 'three_with_two',
  FOUR_WITH_ONE: 'four_with_one',
  STRAIGHT_FLUSH: 'straight_flush',
  FIVE_OF_KIND: 'five_of_kind'
};

const CARD_TYPE_ORDER = {
  five_of_kind: 9,
  straight_flush: 8,
  four_with_one: 7,
  three_with_two: 6,
  flush: 5,
  straight: 4,
  triple: 3,
  pair: 2,
  single: 1
};

function createSingleDeck(deckIndex) {
  const deck = [];
  let id = 0;
  const suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS];
  const ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
  for (const suit of suits) {
    for (const rank of ranks) {
      deck.push({ id: `deck${deckIndex}-card-${id++}`, suit, rank, isJoker: false, isBigJoker: false, isSmallJoker: false });
    }
  }
  deck.push({ id: `deck${deckIndex}-card-${id++}`, suit: Suit.JOKER_BLACK, rank: 16, isJoker: true, isBigJoker: false, isSmallJoker: true });
  deck.push({ id: `deck${deckIndex}-card-${id++}`, suit: Suit.JOKER_RED, rank: RANK_JOKER_BIG, isJoker: true, isBigJoker: true, isSmallJoker: false });
  return deck;
}

export function createDeck() {
  const deck = [];
  for (let d = 0; d < 3; d++) deck.push(...createSingleDeck(d));
  return deck;
}

export function shuffleDeck(deck) {
  const shuffled = [...deck];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

export function dealCards(deck, playerCount) {
  const hands = Array.from({ length: playerCount }, () => []);
  const cardsPerPlayer = Math.floor(deck.length / playerCount);
  for (let i = 0; i < cardsPerPlayer * playerCount; i++) hands[i % playerCount].push(deck[i]);
  return hands;
}

export function getCardSortValue(card, trumpRank) {
  if (card.isBigJoker) return 100;
  if (card.isSmallJoker) return 99;
  const rank = card.rank;
  if (trumpRank != null && rank === trumpRank) return 98;
  return rank;
}

export function sortCards(cards, jokers, trumpRank) {
  return [...cards].sort((a, b) => getCardSortValue(b, trumpRank) - getCardSortValue(a, trumpRank));
}

function isTrump(card, jokers, trumpRank) {
  if (card.isJoker) return true;
  if (trumpRank != null && card.rank === trumpRank) return true;
  return false;
}

function groupByRank(cards) {
  const groups = {};
  for (const card of cards) {
    const key = card.rank;
    if (!groups[key]) groups[key] = [];
    groups[key].push(card);
  }
  return groups;
}

function getRankValue(rank, trumpRank) {
  if (rank === RANK_JOKER_BIG) return 100;
  if (rank === 16) return 99;
  if (trumpRank != null && rank === trumpRank) return 98;
  return rank;
}

// 辅助函数：分离王牌和普通牌
function separateJokers(cards) {
  const jokers = cards.filter(c => c.isJoker);
  const nonJokers = cards.filter(c => !c.isJoker);
  return { jokers, nonJokers, jokerCount: jokers.length };
}

// 辅助函数：获取普通牌的点数统计
function getRankCounts(nonJokers) {
  const counts = {};
  for (const card of nonJokers) {
    counts[card.rank] = (counts[card.rank] || 0) + 1;
  }
  return counts;
}

// 辅助函数：尝试用王牌组成五同（优先级最高）
// 规则：同一牌型中点数从小，但五同是最强牌型
function tryFiveOfKind(nonJokers, jokerCount, trumpRank) {
  const counts = getRankCounts(nonJokers);
  // 找到所有可以组成五同的点数
  // 规则：优先选择能组成五同的点数（五同是最强牌型）
  let bestRank = null;
  for (const [rank, count] of Object.entries(counts)) {
    const needed = 5 - count;
    if (jokerCount >= needed) {
      const r = parseInt(rank);
      if (bestRank === null || r > bestRank) {
        bestRank = r;
      }
    }
  }
  if (bestRank !== null) {
    return { type: CardType.FIVE_OF_KIND, mainRank: bestRank, length: 5 };
  }
  // 如果全是王牌
  if (nonJokers.length === 0 && jokerCount >= 5) {
    // 大王最大
    return { type: CardType.FIVE_OF_KIND, mainRank: RANK_JOKER_BIG, length: 5 };
  }
  return null;
}

// 辅助函数：尝试用王牌组成四带一
// 规则：同一牌型中点数从小，四带一作为四张的点数要尽量小
// 注意：将牌是最大的，所以"从小"意味着尽量不选将牌
function tryFourWithOne(nonJokers, jokerCount, trumpRank) {
  const counts = getRankCounts(nonJokers);
  
  let bestRank = null;
  let bestRankOrder = null;
  
  // 尝试找四张相同的点数（用王牌补充）
  for (const [rank, count] of Object.entries(counts)) {
    const needed = 4 - count;
    if (jokerCount >= needed) {
      // 需要检查剩余的牌是否能作为"一"
      const usedJokers = needed;
      const remainingJokers = jokerCount - usedJokers;
      const otherCards = nonJokers.filter(c => c.rank !== parseInt(rank));
      // 其他牌或剩余王牌可以作为"一"
      if (otherCards.length + remainingJokers >= 1) {
        const r = parseInt(rank);
        const order = getRankOrderForMinRule(r, trumpRank);
        if (bestRankOrder === null || order < bestRankOrder) {
          bestRank = r;
          bestRankOrder = order;
        }
      }
    }
  }
  
  if (bestRank !== null) {
    return { type: CardType.FOUR_WITH_ONE, mainRank: bestRank, length: 5 };
  }
  return null;
}

// 辅助函数：获取牌的实际大小值（用于"点数从小"规则）
// 注意：将牌是最大的，所以在比较"谁更小"时，将牌应该被视为最大
function getRankOrderForMinRule(rank, trumpRank) {
  if (rank === RANK_JOKER_BIG) return 100;  // 大王最大
  if (rank === 16) return 99;  // 小王第二大
  if (trumpRank != null && rank === trumpRank) return 98;  // 将牌第三大
  return rank;  // 普通牌按点数
}

// 辅助函数：尝试用王牌组成三带两
// 规则：同一牌型中点数从小，三张的点数要尽量小
// 注意：将牌是最大的，所以"从小"意味着尽量不选将牌
// 例如：王+22+33（将牌是2）→ 王+22变333+22（三张3），而不是222+33（三张是将牌2）
// 例如：6633+王（将牌不是6也不是3）→ 33366（王充当3），而不是 66633
function tryThreeWithTwo(nonJokers, jokerCount, trumpRank) {
  const counts = getRankCounts(nonJokers);
  
  // 找到所有可以组成三带两的点数，选择最小的
  // 注意：将牌是最大的，所以在"从小"规则中，将牌应该被视为最大的
  let bestRank = null;
  let bestRankOrder = null;  // 用于比较的实际顺序值
  
  // 尝试找三张相同的点数（用王牌补充）
  for (const [rank, count] of Object.entries(counts)) {
    const neededForTriple = 3 - count;
    if (jokerCount >= neededForTriple) {
      const usedJokers = neededForTriple;
      const remainingJokers = jokerCount - usedJokers;
      const otherCards = nonJokers.filter(c => c.rank !== parseInt(rank));
      
      // 检查剩余牌能否组成对子
      const otherCounts = getRankCounts(otherCards);
      let canFormPair = false;
      for (const [otherRank, otherCount] of Object.entries(otherCounts)) {
        const neededForPair = 2 - otherCount;
        if (remainingJokers >= neededForPair) {
          canFormPair = true;
          break;
        }
      }
      
      if (canFormPair) {
        const r = parseInt(rank);
        const order = getRankOrderForMinRule(r, trumpRank);
        // 规则：同一牌型中点数从小，选择"最小"的点数
        // 注意：将牌order值最大（98），所以不会被优先选择
        if (bestRankOrder === null || order < bestRankOrder) {
          bestRank = r;
          bestRankOrder = order;
        }
      }
    }
  }
  
  if (bestRank !== null) {
    return { type: CardType.THREE_WITH_TWO, mainRank: bestRank, length: 5 };
  }
  return null;
}

// 辅助函数：尝试用王牌组成同花顺
// 规则：同花顺的点数从最大值开始（牌型本身就是从大到小比较）
function tryStraightFlush(nonJokers, jokerCount, trumpRank) {
  if (nonJokers.length === 0) return null;
  
  // 检查是否所有非王牌都是同花色
  const suits = [...new Set(nonJokers.map(c => c.suit))];
  if (suits.length > 1) return null;
  
  const suit = suits[0];
  const ranks = nonJokers.map(c => c.rank).sort((a, b) => a - b);
  
  // 尝试各种可能的顺子，从最小的起点开始
  // 规则：同一牌型中点数从小，所以选择最小起点的顺子
  for (let start = 2; start <= 10; start++) {
    const straightRanks = [start, start + 1, start + 2, start + 3, start + 4];
    let needed = 0;
    for (const r of straightRanks) {
      if (!ranks.includes(r)) needed++;
    }
    if (needed <= jokerCount) {
      // 返回第一个能组成的顺子（起点最小）
      return { type: CardType.STRAIGHT_FLUSH, mainRank: start + 4, length: 5 };
    }
  }
  return null;
}

// 辅助函数：尝试用王牌组成同花
// 规则：同花的点数从最大值开始（牌型本身就是从大到小比较）
function tryFlush(nonJokers, jokerCount, trumpRank) {
  if (nonJokers.length === 0 && jokerCount < 5) return null;
  
  // 检查是否所有非王牌都是同花色
  const suits = [...new Set(nonJokers.map(c => c.suit))];
  if (suits.length > 1) return null;
  
  if (nonJokers.length === 0) {
    // 全是王牌，可以组成同花
    return { type: CardType.FLUSH, mainRank: RANK_JOKER_BIG, length: 5 };
  }
  
  // 有王牌补充，可以组成同花
  return { type: CardType.FLUSH, mainRank: Math.max(...nonJokers.map(c => c.rank)), length: 5 };
}

// 辅助函数：尝试用王牌组成顺子
// 规则：同一牌型中点数从小，所以选择最小起点的顺子
// 例如：3456+王 → 23456（起点是2），而不是 34567
function tryStraight(nonJokers, jokerCount, trumpRank) {
  if (nonJokers.length === 0 && jokerCount < 5) return null;
  
  const ranks = nonJokers.map(c => c.rank).sort((a, b) => a - b);
  
  // 尝试各种可能的顺子，从最小的起点开始
  for (let start = 2; start <= 10; start++) {
    const straightRanks = [start, start + 1, start + 2, start + 3, start + 4];
    let needed = 0;
    for (const r of straightRanks) {
      if (!ranks.includes(r)) needed++;
    }
    if (needed <= jokerCount) {
      // 返回第一个能组成的顺子（起点最小）
      return { type: CardType.STRAIGHT, mainRank: start + 4, length: 5 };
    }
  }
  return null;
}

// 辅助函数：尝试用王牌组成三张
// 规则：同一牌型中点数从小，选择最小的点数
// 注意：将牌是最大的，所以"从小"意味着尽量不选将牌
function tryTriple(nonJokers, jokerCount, trumpRank) {
  const counts = getRankCounts(nonJokers);
  
  let bestRank = null;
  let bestRankOrder = null;
  
  for (const [rank, count] of Object.entries(counts)) {
    const needed = 3 - count;
    if (jokerCount >= needed) {
      const r = parseInt(rank);
      const order = getRankOrderForMinRule(r, trumpRank);
      if (bestRankOrder === null || order < bestRankOrder) {
        bestRank = r;
        bestRankOrder = order;
      }
    }
  }
  
  // 全是王牌
  if (nonJokers.length === 0 && jokerCount >= 3) {
    bestRank = RANK_JOKER_BIG;
  }
  
  if (bestRank !== null) {
    return { type: CardType.TRIPLE, mainRank: bestRank, length: 3 };
  }
  return null;
}

// 辅助函数：尝试用王牌组成对子
// 规则：同一牌型中点数从小，选择最小的点数
// 注意：将牌是最大的，所以"从小"意味着尽量不选将牌
function tryPair(nonJokers, jokerCount, trumpRank) {
  const counts = getRankCounts(nonJokers);
  
  let bestRank = null;
  let bestRankOrder = null;
  
  for (const [rank, count] of Object.entries(counts)) {
    const needed = 2 - count;
    if (jokerCount >= needed) {
      const r = parseInt(rank);
      const order = getRankOrderForMinRule(r, trumpRank);
      if (bestRankOrder === null || order < bestRankOrder) {
        bestRank = r;
        bestRankOrder = order;
      }
    }
  }
  
  // 全是王牌
  if (nonJokers.length === 0 && jokerCount >= 2) {
    bestRank = RANK_JOKER_BIG;
  }
  
  if (bestRank !== null) {
    return { type: CardType.PAIR, mainRank: bestRank, length: 2 };
  }
  return null;
}

export function recognizeCardType(cards, jokers, trumpRank) {
  if (!cards || cards.length === 0) return null;
  const n = cards.length;

  // 单张
  if (n === 1) {
    return { type: CardType.SINGLE, mainRank: cards[0].rank, length: 1 };
  }

  // 分离王牌和普通牌
  const { jokers: jokerCards, nonJokers, jokerCount } = separateJokers(cards);

  // 对子（2张）
  if (n === 2) {
    const result = tryPair(nonJokers, jokerCount, trumpRank);
    return result;
  }

  // 三张（3张）
  if (n === 3) {
    const result = tryTriple(nonJokers, jokerCount, trumpRank);
    return result;
  }

  // 5张牌型 - 按牌型优先级从高到低尝试
  if (n === 5) {
    // 1. 尝试五同（最大）
    const fiveOfKind = tryFiveOfKind(nonJokers, jokerCount, trumpRank);
    if (fiveOfKind) return fiveOfKind;

    // 2. 尝试同花顺
    const straightFlush = tryStraightFlush(nonJokers, jokerCount, trumpRank);
    if (straightFlush) return straightFlush;

    // 3. 尝试四带一
    const fourWithOne = tryFourWithOne(nonJokers, jokerCount, trumpRank);
    if (fourWithOne) return fourWithOne;

    // 4. 尝试三带两
    const threeWithTwo = tryThreeWithTwo(nonJokers, jokerCount, trumpRank);
    if (threeWithTwo) return threeWithTwo;

    // 5. 尝试同花
    const flush = tryFlush(nonJokers, jokerCount, trumpRank);
    if (flush) return flush;

    // 6. 尝试顺子
    const straight = tryStraight(nonJokers, jokerCount, trumpRank);
    if (straight) return straight;
  }

  return null;
}

export function compareCards(played, current, jokers, trumpRank) {
  if (!played || !current) return 0;
  const pt = recognizeCardType(played, jokers, trumpRank);
  const ct = recognizeCardType(current, jokers, trumpRank);
  if (!pt || !ct) return 0;
  if (pt.length !== ct.length) return 0;
  if (CARD_TYPE_ORDER[pt.type] !== CARD_TYPE_ORDER[ct.type]) {
    return CARD_TYPE_ORDER[ct.type] > CARD_TYPE_ORDER[pt.type] ? 1 : -1;
  }
  const pvMain = getRankValue(pt.mainRank, trumpRank);
  const cvMain = getRankValue(ct.mainRank, trumpRank);
  if (cvMain > pvMain) return 1;
  if (cvMain < pvMain) return -1;
  return 0;
}

export function countJokersInHand(hand) {
  return hand.filter(c => c.isJoker).length;
}

export function getJokerCards(hand) {
  return hand.filter(c => c.isJoker);
}

export function getBestCard(hand, jokers) {
  if (!hand || hand.length === 0) return null;
  return sortCards(hand, jokers)[0];
}

export function getBestCards(hand, jokers) {
  if (!hand || hand.length === 0) return [];
  const sorted = sortCards(hand, jokers);
  const best = sorted[0];
  if (!best) return [];
  if (best.isJoker) return [best];
  return sorted.filter(c => c.rank === best.rank && !c.isJoker);
}

// AI 自动选牌（增强版：支持王牌作为自由牌）
export function aiSelectCards(hand, jokers, currentPlayed, isFirstPlay, requiredCount, trumpRank) {
  const sorted = sortCards(hand, jokers, trumpRank);

  if (isFirstPlay || !currentPlayed || currentPlayed.length === 0) {
    // 首发：枚举所有合法牌型（1/2/3/5张），随机选一个
    const allOptions = findAllFirstPlayOptions(hand, jokers, trumpRank);
    if (allOptions.length > 0) {
      return allOptions[Math.floor(Math.random() * allOptions.length)];
    }
    // 兜底：出最小的单张
    return [sorted[sorted.length - 1]];
  }
  
  // 跟牌：找能压住的最小组合
  const needed = currentPlayed.length;
  const currentType = recognizeCardType(currentPlayed, jokers, trumpRank);
  if (!currentType) return [];
  
  // 尝试找到能压住的最小组合
  const candidates = findAllValidCombinations(hand, needed, currentType, trumpRank);
  if (candidates.length === 0) return [];
  
  // 返回点数最小的组合
  candidates.sort((a, b) => {
    const typeA = recognizeCardType(a, jokers, trumpRank);
    const typeB = recognizeCardType(b, jokers, trumpRank);
    if (!typeA || !typeB) return 0;
    // 先按牌型排序（同一牌型）
    if (typeA.type !== typeB.type) {
      return CARD_TYPE_ORDER[typeA.type] - CARD_TYPE_ORDER[typeB.type];
    }
    // 同一牌型按点数从小到大
    return typeA.mainRank - typeB.mainRank;
  });
  
  // 找到第一个能压住当前牌的组合
  for (const candidate of candidates) {
    if (compareCards(currentPlayed, candidate, jokers, trumpRank) === 1) {
      return candidate;
    }
  }
  
  return []; // 不出
}

// 枚举所有首发合法牌型（1/2/3/5张），去重后返回
function findAllFirstPlayOptions(hand, jokers, trumpRank) {
  const nonJokers = hand.filter(c => !c.isJoker);
  const jokerCards = hand.filter(c => c.isJoker);
  const jc = jokerCards.length;
  const counts = getRankCounts(nonJokers);
  const options = [];
  const seen = new Set();

  function add(cards) {
    const type = recognizeCardType(cards, jokers, trumpRank);
    if (!type) return;
    const key = `${type.type}-${type.mainRank}`;
    if (!seen.has(key)) {
      seen.add(key);
      options.push(cards);
    }
  }

  // ── 单张 ──
  for (const card of nonJokers) {
    add([card]);
  }
  for (const j of jokerCards.slice(0, 1)) {
    add([j]);
  }

  // ── 对子 ──
  for (const [rank, cnt] of Object.entries(counts)) {
    const r = parseInt(rank);
    const rankCards = nonJokers.filter(c => c.rank === r);
    if (cnt >= 2) {
      add(rankCards.slice(0, 2));
    } else if (jc >= 1) {
      add([rankCards[0], jokerCards[0]]);
    }
  }
  if (jc >= 2) add(jokerCards.slice(0, 2));

  // ── 三张 ──
  for (const [rank, cnt] of Object.entries(counts)) {
    const r = parseInt(rank);
    const rankCards = nonJokers.filter(c => c.rank === r);
    const need = Math.max(0, 3 - cnt);
    if (jc >= need) {
      add([...rankCards.slice(0, 3 - need), ...jokerCards.slice(0, need)]);
    }
  }
  if (jc >= 3) add(jokerCards.slice(0, 3));

  // ── 5张：所有牌型 ──
  if (hand.length >= 5) {
    // 顺子
    for (let start = 2; start <= 10; start++) {
      const combo = [];
      let usedJ = 0;
      let ok = true;
      for (let r = start; r <= start + 4; r++) {
        const card = nonJokers.find(c => c.rank === r && !combo.includes(c));
        if (card) {
          combo.push(card);
        } else if (usedJ < jc) {
          combo.push(jokerCards[usedJ++]);
        } else {
          ok = false; break;
        }
      }
      if (ok) add(combo);
    }

    // 同花顺
    const suitGroups = {};
    for (const c of nonJokers) {
      if (!suitGroups[c.suit]) suitGroups[c.suit] = [];
      suitGroups[c.suit].push(c);
    }
    for (const [, sc] of Object.entries(suitGroups)) {
      for (let start = 2; start <= 10; start++) {
        const combo = [];
        let usedJ = 0;
        let ok = true;
        for (let r = start; r <= start + 4; r++) {
          const card = sc.find(c => c.rank === r && !combo.includes(c));
          if (card) {
            combo.push(card);
          } else if (usedJ < jc) {
            combo.push(jokerCards[usedJ++]);
          } else {
            ok = false; break;
          }
        }
        if (ok) add(combo);
      }
    }

    // 同花（非顺子）
    for (const [, sc] of Object.entries(suitGroups)) {
      if (sc.length >= 5) {
        const sorted5 = [...sc].sort((a, b) => a.rank - b.rank);
        add(sorted5.slice(0, 5));
      }
    }

    // 三带两
    for (const [rank, cnt] of Object.entries(counts)) {
      const r = parseInt(rank);
      const threeCards = nonJokers.filter(c => c.rank === r);
      const need3 = Math.max(0, 3 - cnt);
      if (jc < need3) continue;
      const three = [...threeCards.slice(0, 3 - need3), ...jokerCards.slice(0, need3)];
      const remJ = jokerCards.slice(need3);
      for (const [rank2, cnt2] of Object.entries(counts)) {
        if (rank2 === rank) continue;
        const r2 = parseInt(rank2);
        const pairCards = nonJokers.filter(c => c.rank === r2);
        const need2 = Math.max(0, 2 - cnt2);
        if (remJ.length >= need2) {
          add([...three, ...pairCards.slice(0, 2 - need2), ...remJ.slice(0, need2)]);
          break;
        }
      }
      if (remJ.length >= 2) add([...three, ...remJ.slice(0, 2)]);
    }

    // 四带一（优先带小单张）
    for (const [rank, cnt] of Object.entries(counts)) {
      const r = parseInt(rank);
      const fourCards = nonJokers.filter(c => c.rank === r);
      const need4 = Math.max(0, 4 - cnt);
      if (jc < need4) continue;
      const four = [...fourCards.slice(0, 4 - need4), ...jokerCards.slice(0, need4)];
      const remJ = jokerCards.slice(need4);
      
      // 收集所有可以带的单牌，按点数从小到大排序
      const singleCards = [];
      for (const [rank2, cnt2] of Object.entries(counts)) {
        if (rank2 === rank) continue;
        const r2 = parseInt(rank2);
        // 优先带单张（cnt2=1），其次带对子拆牌
        const card1 = nonJokers.find(c => c.rank === r2);
        if (card1) {
          singleCards.push({ card: card1, rank: r2, isSingle: cnt2 === 1 });
        }
      }
      // 排序：单张优先，然后按点数从小到大
      singleCards.sort((a, b) => {
        // 单张优先
        if (a.isSingle !== b.isSingle) return a.isSingle ? -1 : 1;
        // 点数从小到大（王牌视为最大）
        const aVal = a.card.isJoker ? 100 : a.rank;
        const bVal = b.card.isJoker ? 100 : b.rank;
        return aVal - bVal;
      });
      
      if (singleCards.length > 0) {
        add([...four, singleCards[0].card]);
      } else if (remJ.length >= 1) {
        add([...four, remJ[0]]);
      }
    }

    // 五同
    for (const [rank, cnt] of Object.entries(counts)) {
      const r = parseInt(rank);
      const rankCards = nonJokers.filter(c => c.rank === r);
      const need = Math.max(0, 5 - cnt);
      if (jc >= need) {
        add([...rankCards.slice(0, 5 - need), ...jokerCards.slice(0, need)]);
      }
    }
    if (jc >= 5) add(jokerCards.slice(0, 5));
  }

  return options;
}

// 找到最小的5张牌组合
function findSmallestFiveCardHand(hand, trumpRank) {
  const sorted = sortCards(hand, null, trumpRank);
  const jokers = hand.filter(c => c.isJoker);
  const nonJokers = sorted.filter(c => !c.isJoker);
  
  // 优先级：顺子 > 同花 > 三带两 > 四带一 > 五同
  // 但我们想要最小的，所以反过来
  
  // 1. 尝试顺子（最小点数）
  for (let start = 2; start <= 10; start++) {
    const straightRanks = [start, start + 1, start + 2, start + 3, start + 4];
    const cards = [];
    let needed = 0;
    for (const r of straightRanks) {
      const card = nonJokers.find(c => c.rank === r && !cards.includes(c));
      if (card) {
        cards.push(card);
      } else {
        needed++;
      }
    }
    if (needed <= jokers.length) {
      // 补充王牌
      const jokersToAdd = jokers.slice(0, needed);
      return [...cards, ...jokersToAdd];
    }
  }
  
  // 2. 尝试同花（最小点数）
  const suitGroups = {};
  for (const card of nonJokers) {
    if (!suitGroups[card.suit]) suitGroups[card.suit] = [];
    suitGroups[card.suit].push(card);
  }
  for (const [suit, cards] of Object.entries(suitGroups)) {
    if (cards.length + jokers.length >= 5) {
      const needed = 5 - cards.length;
      if (needed <= jokers.length) {
        // 选择点数最小的5张
        cards.sort((a, b) => a.rank - b.rank);
        const jokersToAdd = jokers.slice(0, needed);
        return [...cards.slice(0, 5 - needed), ...jokersToAdd];
      }
    }
  }
  
  // 3. 尝试三带两（最小点数）
  const counts = getRankCounts(nonJokers);
  let minTripleRank = null;
  for (const [rank, cnt] of Object.entries(counts)) {
    const needed = 3 - cnt;
    if (jokers.length >= needed) {
      const r = parseInt(rank);
      if (minTripleRank === null || r < minTripleRank) minTripleRank = r;
    }
  }
  if (minTripleRank !== null) {
    const tripleCards = nonJokers.filter(c => c.rank === minTripleRank);
    const neededForTriple = 3 - tripleCards.length;
    // 找最小的对子作为"两"
    let minPairRank = null;
    for (const [rank, cnt] of Object.entries(counts)) {
      if (parseInt(rank) !== minTripleRank) {
        const neededForPair = 2 - cnt;
        const remainingJokers = jokers.length - neededForTriple;
        if (remainingJokers >= neededForPair) {
          const r = parseInt(rank);
          if (minPairRank === null || r < minPairRank) minPairRank = r;
        }
      }
    }
    if (minPairRank !== null) {
      const pairCards = nonJokers.filter(c => c.rank === minPairRank);
      const tripleJokers = jokers.slice(0, neededForTriple);
      const remainingJokers = jokers.slice(neededForTriple);
      const neededForPair = 2 - pairCards.length;
      const pairJokers = remainingJokers.slice(0, neededForPair);
      return [...tripleCards, ...tripleJokers, ...pairCards, ...pairJokers];
    }
  }
  
  return null;
}

// 找到所有有效的牌型组合
function findAllValidCombinations(hand, count, targetType, trumpRank) {
  const jokers = hand.filter(c => c.isJoker);
  const nonJokers = hand.filter(c => !c.isJoker);
  const candidates = [];
  
  if (count === 1) {
    // 单张
    for (const card of hand) {
      candidates.push([card]);
    }
  } else if (count === 2) {
    // 对子
    const counts = getRankCounts(nonJokers);
    for (const [rank, cnt] of Object.entries(counts)) {
      const r = parseInt(rank);
      const cards = nonJokers.filter(c => c.rank === r);
      if (cnt >= 2) {
        candidates.push(cards.slice(0, 2));
      }
      // 用王牌补充
      if (cnt >= 1 && jokers.length >= 1) {
        candidates.push([cards[0], jokers[0]]);
      }
    }
    if (jokers.length >= 2) {
      candidates.push(jokers.slice(0, 2));
    }
  } else if (count === 3) {
    // 三张
    const counts = getRankCounts(nonJokers);
    for (const [rank, cnt] of Object.entries(counts)) {
      const r = parseInt(rank);
      const cards = nonJokers.filter(c => c.rank === r);
      if (cnt >= 3) {
        candidates.push(cards.slice(0, 3));
      }
      // 用王牌补充
      const needed = 3 - cnt;
      if (jokers.length >= needed) {
        candidates.push([...cards, ...jokers.slice(0, needed)]);
      }
    }
    if (jokers.length >= 3) {
      candidates.push(jokers.slice(0, 3));
    }
  } else if (count === 5) {
    // 5张牌型
    // 五同
    const fiveOfKind = tryFiveOfKind(nonJokers, jokers.length, trumpRank);
    if (fiveOfKind) {
      const cards = buildFiveOfKind(nonJokers, jokers, fiveOfKind.mainRank);
      if (cards) candidates.push(cards);
    }
    
    // 同花顺
    const straightFlush = tryStraightFlush(nonJokers, jokers.length, trumpRank);
    if (straightFlush) {
      const cards = buildStraightFlush(nonJokers, jokers, straightFlush.mainRank);
      if (cards) candidates.push(cards);
    }
    
    // 四带一
    const fourWithOne = tryFourWithOne(nonJokers, jokers.length, trumpRank);
    if (fourWithOne) {
      const cards = buildFourWithOne(nonJokers, jokers, fourWithOne.mainRank);
      if (cards) candidates.push(cards);
    }
    
    // 三带两
    const threeWithTwo = tryThreeWithTwo(nonJokers, jokers.length, trumpRank);
    if (threeWithTwo) {
      const cards = buildThreeWithTwo(nonJokers, jokers, threeWithTwo.mainRank);
      if (cards) candidates.push(cards);
    }
    
    // 同花
    const flush = tryFlush(nonJokers, jokers.length, trumpRank);
    if (flush) {
      const cards = buildFlush(nonJokers, jokers, flush.mainRank);
      if (cards) candidates.push(cards);
    }
    
    // 顺子
    const straight = tryStraight(nonJokers, jokers.length, trumpRank);
    if (straight) {
      const cards = buildStraight(nonJokers, jokers, straight.mainRank);
      if (cards) candidates.push(cards);
    }
  }
  
  // 过滤掉不符合目标牌型的组合
  return candidates.filter(cards => {
    const type = recognizeCardType(cards, jokers, trumpRank);
    return type && type.type === targetType.type;
  });
}

// 构建五同
function buildFiveOfKind(nonJokers, jokers, mainRank) {
  const cards = nonJokers.filter(c => c.rank === mainRank);
  const needed = 5 - cards.length;
  if (jokers.length < needed) return null;
  return [...cards, ...jokers.slice(0, needed)];
}

// 构建同花顺
function buildStraightFlush(nonJokers, jokers, mainRank) {
  const start = mainRank - 4;
  const cards = [];
  const jokersUsed = [];
  let jokerIndex = 0;
  
  for (let r = start; r <= mainRank; r++) {
    const card = nonJokers.find(c => c.rank === r && !cards.includes(c));
    if (card) {
      cards.push(card);
    } else if (jokerIndex < jokers.length) {
      cards.push(jokers[jokerIndex++]);
    } else {
      return null;
    }
  }
  return cards;
}

// 构建四带一（优先带小单张）
function buildFourWithOne(nonJokers, jokers, mainRank) {
  const fourCards = nonJokers.filter(c => c.rank === mainRank);
  const neededForFour = 4 - fourCards.length;
  if (jokers.length < neededForFour) return null;
  
  const fourWithJokers = [...fourCards, ...jokers.slice(0, neededForFour)];
  
  // 找一张最小的牌作为"一"，优先选单张
  const remainingNonJokers = nonJokers.filter(c => c.rank !== mainRank);
  const remainingJokers = jokers.slice(neededForFour);
  
  if (remainingNonJokers.length > 0) {
    // 统计剩余牌的数量，优先选单张
    const counts = getRankCounts(remainingNonJokers);
    const candidates = remainingNonJokers.map(c => ({
      card: c,
      isSingle: counts[c.rank] === 1
    }));
    
    // 排序：单张优先，然后按点数从小到大
    candidates.sort((a, b) => {
      if (a.isSingle !== b.isSingle) return a.isSingle ? -1 : 1;
      return a.card.rank - b.card.rank;
    });
    
    return [...fourWithJokers, candidates[0].card];
  } else if (remainingJokers.length > 0) {
    return [...fourWithJokers, remainingJokers[0]];
  }
  return null;
}

// 构建三带两
function buildThreeWithTwo(nonJokers, jokers, mainRank) {
  const threeCards = nonJokers.filter(c => c.rank === mainRank);
  const neededForThree = 3 - threeCards.length;
  if (jokers.length < neededForThree) return null;
  
  const threeWithJokers = [...threeCards, ...jokers.slice(0, neededForThree)];
  
  // 找最小的对子作为"两"
  const remainingNonJokers = nonJokers.filter(c => c.rank !== mainRank);
  const remainingJokers = jokers.slice(neededForThree);
  
  const counts = getRankCounts(remainingNonJokers);
  let minPairRank = null;
  for (const [rank, cnt] of Object.entries(counts)) {
    const neededForPair = 2 - cnt;
    if (remainingJokers.length >= neededForPair) {
      const r = parseInt(rank);
      if (minPairRank === null || r < minPairRank) minPairRank = r;
    }
  }
  
  if (minPairRank !== null) {
    const pairCards = remainingNonJokers.filter(c => c.rank === minPairRank);
    const neededForPair = 2 - pairCards.length;
    const pairJokers = remainingJokers.slice(0, neededForPair);
    return [...threeWithJokers, ...pairCards, ...pairJokers];
  }
  return null;
}

// 构建同花
function buildFlush(nonJokers, jokers, mainRank) {
  const suitGroups = {};
  for (const card of nonJokers) {
    if (!suitGroups[card.suit]) suitGroups[card.suit] = [];
    suitGroups[card.suit].push(card);
  }
  
  for (const [suit, cards] of Object.entries(suitGroups)) {
    if (cards.length + jokers.length >= 5) {
      const needed = 5 - cards.length;
      if (needed <= jokers.length) {
        // 选择点数最小的5张
        cards.sort((a, b) => a.rank - b.rank);
        return [...cards.slice(0, 5 - needed), ...jokers.slice(0, needed)];
      }
    }
  }
  return null;
}

// 构建顺子
function buildStraight(nonJokers, jokers, mainRank) {
  const start = mainRank - 4;
  const cards = [];
  let jokerIndex = 0;
  
  for (let r = start; r <= mainRank; r++) {
    const card = nonJokers.find(c => c.rank === r && !cards.includes(c));
    if (card) {
      cards.push(card);
    } else if (jokerIndex < jokers.length) {
      cards.push(jokers[jokerIndex++]);
    } else {
      return null;
    }
  }
  return cards;
}

export function getCardDisplay(card) {
  if (card.isBigJoker) return '大王';
  if (card.isSmallJoker) return '小王';
  const rankMap = { 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K', 14: 'A' };
  return `${card.suit}${rankMap[card.rank] || card.rank}`;
}
