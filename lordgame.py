#!/usr/bin/python3

from enum import Enum
from model import LordAgent
import random


class CardSuit(Enum):
    ClubSuit = 1
    DiamondSuit = 2
    SpadeSuit = 3
    HeartSuit = 4


def valid_card(value: int):
    return value >= 3 and value <= 16


def card_compare(c1, c2) -> int:
    assert(valid_card(c1.card_value) and valid_card(c2.card_value))
    if c1.card_value == c2.card_value:
        return 0
    elif c1.card_value < c2.card_value:
        return -1
    else:
        return 1


class CardColor:
    CLUBCOLOR = '\033[38;2;150;150;200m'
    HEARTCOLOR = '\033[38;2;255;0;0m'
    DIAMONDCOLOR = '\033[38;2;50;200;50m'
    SPADECOLOR = '\033[38;2;100;100;100m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


def card_to_value(c) -> str:
    assert(valid_card(c.card_value))
    ret: str = ""
    if c.card_suit == CardSuit.ClubSuit:
        ret += CardColor.CLUBCOLOR
    elif c.card_suit == CardSuit.DiamondSuit:
        ret += CardColor.DIAMONDCOLOR
    elif c.card_suit == CardSuit.HeartSuit:
        ret += CardColor.HEARTCOLOR
    elif c.card_suit == CardSuit.SpadeSuit:
        ret += CardColor.SPADECOLOR

    if c.card_value == 11:
        ret += "J "
    elif c.card_value == 12:
        ret += "Q "
    elif c.card_value == 13:
        ret += "K "
    elif c.card_value == 14:
        ret += "A "
    elif c.card_value == 15:
        ret += "2 "
    elif c.card_value == 16:
        ret += "$ "
    elif c.card_value == 10:
        ret += "\u2469 "
    else:
        ret += str(c.card_value)
        ret += " "

    ret += CardColor.ENDC
    return ret


class Card():
    def __init__(self, value: int, suit: CardSuit):
        if not valid_card(value):
            raise "bad card value"
        self.card_value = value
        self.card_suit = suit

    def __gt__(self, other):
        return card_compare(self, other) == 1

    def __lt__(self, other):
        return card_compare(self, other) == -1

    def __eq__(self, other):
        return card_compare(self, other) == 0

    def __str__(self):
        return card_to_value(self)


def card_sort(c: Card):
    return c.card_value * 50 + c.card_suit.value


def all_eq(l: []) -> bool:
    if l.__len__() == 0:
        return True
    l1 = l[0]
    for v in l:
        if v != l1:
            return False
    return True


class CardCombinationType(Enum):
    Invalid = 0
    Single = 1
    Double = 2
    Three = 3
    ThreeA = 4
    ThreeB = 5
    Four = 6
    Straight = 7
    MultiDouble = 8
    MultiThree = 9
    MultiThreeA = 10
    MultiThreeB = 11
    FourAndTwo = 12


class CardCombination():
    def __init__(self, *argv: [Card]):
        self.__store: [Card] = []
        for arg in argv:
            self.__store.append(arg)
        self.__store.sort(key=card_sort)

    def AddCard(self, *cards: [Card]):
        for card in cards:
            self.__store.append(card)
        self.__store.sort(key=card_sort)
        return self

    def Cards(self) -> [Card]:
        return self.__store.copy()

    def copy(self):
        return CardCombination(*self.__store.copy())

    def __test_straight(self) -> CardCombinationType:
        invalid = CardCombinationType.Invalid
        if self.__store.__len__() < 5:
            return invalid
        prev = self.__store[0]
        if prev.card_value < 3 or prev.card_value > 14:
            return invalid
        for i in self.__store[1:]:
            if i.card_value < 3 or i.card_value > 14:
                return invalid
            if i.card_value != prev.card_value + 1:
                return invalid
            prev = i
        return CardCombinationType.Straight

    def __test_multidouble(self) -> CardCombinationType:
        invalid = CardCombinationType.Invalid
        if self.__store.__len__() < 6 or self.__store.__len__() % 2 != 0:
            return invalid
        prev = self.__store[0]
        if prev.card_value < 3 or prev.card_value > 14:
            return invalid
        for i in range(1, self.__store.__len__()):
            c = self.__store[i]
            if c.card_value < 3 or c.card_value > 14:
                return invalid
            if i % 2 == 0:
                if c.card_value != (prev.card_value + 1):
                    return invalid
                prev = c
            else:
                if prev != self.__store[i]:
                    return invalid
        return CardCombinationType.MultiDouble

    def __test_multithree(self) -> CardCombinationType:
        assert(self.__store.__len__() > 0)
        one: [Card] = []
        val: Card = None
        count: int = 0
        for i in self.__store:
            if val is not None and i == val:
                count = count + 1
                if count == 3:
                    one.append(val)
                    val = None
                    count = 0
            else:
                val = i
                count = 1
        if one.__len__() < 1:
            return CardCombinationType.Invalid
        start = one[0]
        prev = one[0]
        for v in one[1:]:
            if v.card_value == prev.card_value + 1:
                prev = v
            else:
                test = self.__test_multithree_finish(start, prev)
                if test != CardCombinationType.Invalid:
                    return test
                start = v
                prev = v
        test = self.__test_multithree_finish(start, prev)
        return test

    def __test_multithree_finish(self, start: Card, end: Card):
        if start.card_value >= 15:
            if self.__store.__len__() == 4 or self.__store.__len__() == 5:
                return self.__type()
            else:
                return CardCombinationType.Invalid
        else:
            if end.card_value >= 15:
                n = 15 - start.card_value + 1
            else:
                n = end.card_value - start.card_value + 1
            if 3 * n == self.__store.__len__():
                return CardCombinationType.MultiThree
            elif 4 * n == self.__store.__len__():
                return CardCombinationType.MultiThreeA
            elif 5 * n == self.__store.__len__():
                a = None
                # FIXME
                for v in self.__store:
                    if(v.card_value >= start.card_value and
                       v.card_value <= end.card_value):
                        continue
                    if a is None:
                        a = v.card_value
                    else:
                        if a != v.card_value:
                            return CardCombinationType.Invalid
                        a = None
                if not (a is None):
                    return CardCombinationType.Invalid
                else:
                    return CardCombinationType.MultiThreeB
            else:
                return CardCombinationType.Invalid

    def __test_fourAndTwo(self) -> CardCombinationType:
        invalid = CardCombinationType.Invalid
        if self.__store.__len__() != 6:
            return invalid
        if(all_eq(self.__store[0:4]) or all_eq(self.__store[1:5]) or
           all_eq(self.__store[2:6])):
            return CardCombinationType.FourAndTwo
        return invalid

    def __type(self) -> CardCombinationType:
        if self.__store.__len__() == 0:
            return CardCombinationType.Invalid
        elif self.__store.__len__() == 1:
            return CardCombinationType.Single
        elif self.__store.__len__() == 2 and all_eq(self.__store):
            return CardCombinationType.Double
        elif self.__store.__len__() == 3 and all_eq(self.__store):
            return CardCombinationType.Three
        elif self.__store.__len__() == 4:
            if all_eq(self.__store):
                return CardCombinationType.Four
            elif all_eq(self.__store[0:3]) or all_eq(self.__store[1:]):
                return CardCombinationType.ThreeA
        elif self.__store.__len__() == 5:
            if((all_eq(self.__store[0:3]) and all_eq(self.__store[3:5])) or
               (all_eq(self.__store[2:5]) and all_eq(self.__store[0:2]))):
                return CardCombinationType.ThreeB
            return self.__test_straight()
        else:
            fourAndTwo = self.__test_fourAndTwo()
            straight = self.__test_straight()
            multiDouble = self.__test_multidouble()
            multiThree = self.__test_multithree()
            if fourAndTwo != CardCombinationType.Invalid:
                return fourAndTwo
            elif straight != CardCombinationType.Invalid:
                return straight
            elif multiDouble != CardCombinationType.Invalid:
                return multiDouble
            elif multiThree != CardCombinationType.Invalid:
                return multiThree
        return CardCombinationType.Invalid

    def cardType(self) -> CardCombinationType:
        return self.__type()

    def valid(self) -> bool:
        return self.cardType() != CardCombinationType.Invalid

    def is_big_brother(self) -> bool:
        if(self.cardType() != CardCombinationType.Double or
           self.__store[0].card_value != 16):
            return False
        return True

    def __sameTypeGreaterThan(self, other) -> bool:
        assert(self.cardType() == other.cardType())
        if self.__store.__len__() != other.__store.__len__():
            return False
        card_type = self.cardType()
        if(card_type == CardCombinationType.Single and
           self.__store[0].card_value == 16 and
           self.__store[0].card_suit == CardSuit.HeartSuit and
           other.__store[0].card_value == 16):
            assert(other.__store[0].card_suit != CardSuit.HeartSuit)
            return True
        if(card_type == CardCombinationType.Single or
           card_type == CardCombinationType.Double or
           card_type == CardCombinationType.Three or
           card_type == CardCombinationType.Four or
           card_type == CardCombinationType.Straight or
           card_type == CardCombinationType.MultiDouble or
           card_type == CardCombinationType.MultiThree):
            return self.__store[0].card_value > other.__store[0].card_value
        a = get_three(self.__store) + get_multiThree(self.__store)
        b = get_three(self.__store) + get_multiThree(other.__store)
        assert(a.__len__() > 0 and b.__len__() > 0)
        am = a[0]
        for i in a:
            if i.__store.__len__() > am.__store.__len__():
                am = i
        bm = b[0]
        for i in b:
            if i.__store.__len__() > bm.__store.__len__():
                bm = i
        return am.__store[0].card_value > bm.__store[0].card_value

    def GreaterThan(self, other) -> bool:
        if(self.cardType() != other.cardType() and
           self.cardType() != CardCombinationType.Four and
           not self.is_big_brother()):
            return False
        if(self.is_big_brother() or
           (self.cardType() == CardCombinationType.Four and
            other.cardType() != CardCombinationType.Four)
           ):
            return True
        return self.__sameTypeGreaterThan(other)

    def __str__(self) -> str:
        ret = ""
        for v in self.__store:
            ret += (str(v) + " ")
        return ret


def get_single(cards: [Card]):
    ret = []
    save = []
    for card in cards:
        if card not in save:
            save.append(card)
            ret.append(CardCombination(card))
    return ret


def get_double(cards: [Card]):
    ret = []
    prev = None
    for card in cards:
        if prev is not None and card == prev:
            ret.append(CardCombination(card, prev))
            prev = None
        else:
            prev = card
    return ret


def get_three(cards: [Card]) -> [CardCombination]:
    ret = []
    prev = None
    count = 0
    for i in range(0, cards.__len__()):
        card = cards[i]
        if prev is not None and card == prev:
            count = count + 1
            if count == 3:
                ret.append(CardCombination(cards[i-2], cards[i-1], card))
                prev = None
                count = 0
        else:
            prev = card
            count = 1
    return ret


def get_four(cards: [Card]) -> [CardCombination]:
    ret = []
    prev = None
    count = 0
    for i in range(0, cards.__len__()):
        card = cards[i]
        if prev is not None and card == prev:
            count = count + 1
            if count == 4:
                ret.append(CardCombination(cards[i-3], cards[i-2],
                                           cards[i-1], card))
                prev = None
                count = 0
        else:
            prev = card
            count = 1
    return ret


def get_threeA(cards: [Card]) -> [CardCombination]:
    ret = []
    threes = get_three(cards)
    for thr in threes:
        assert(thr.cardType() == CardCombinationType.Three)
        for one in get_single(
          [card for card in cards if card not in thr.Cards()]):
            v = thr.copy()
            v.AddCard(*one.Cards())
            ret.append(v)
    return ret


def get_threeB(cards: [Card]) -> [CardCombination]:
    ret = []
    threes = get_three(cards)
    for thr in threes:
        assert(thr.cardType() == CardCombinationType.Three)
        for one in get_double(
          [card for card in cards if card not in thr.Cards()]):
            v = thr.copy()
            v.AddCard(*one.Cards())
            ret.append(v)
    return ret


def get_fourAndTwo(cards: [Card]) -> [CardCombination]:
    ret = []
    fours = get_four(cards)
    for four in fours:
        assert(four.cardType() == CardCombinationType.Four)
        for one1 in get_single(
          [card for card in cards if card not in four.Cards()]):
            for one2 in get_single(
                [card for card in cards if ((card not in four.Cards()) and
                                            (card not in one1.Cards()))]
            ):
                v = four.copy()
                v.AddCard(*one1.Cards(), *one2.Cards())
                ret.append(v)
    return ret
#    return list(dict.fromkeys(ret))


def get_straight_with_long(cards: [Card], long: int) -> [CardCombination]:
    assert(long >= 2)
    ret = []
    scards: [Card] = []
    for c in cards:
        if c in scards:
            continue
        scards.append(c)
    s: Card = None
    p: Card = None
    for i in range(0, scards.__len__()):
        c = scards[i]
        if c.card_value >= 15:
            break
        if s is None or (p.card_value + 1) != c.card_value:
            s = c
            p = c
        else:
            p = c
        assert(p.card_value >= s.card_value)
        if (p.card_value - s.card_value + 1) == long:
            ret.append(CardCombination(*scards[i-long+1:i+1]))
            s = scards[i-long+2]

    return ret


def get_straight(cards: [Card]) -> [CardCombination]:
    ret = []
    for i in range(5, cards.__len__()+1):
        ret += get_straight_with_long(cards, i)
    return ret


def get_multiDouble(cards: [Card]) -> [CardCombination]:
    ret = []
    double_cards: [Card] = []
    prev: Card = None
    count: int = 0
    for c in cards:
        if prev is not None and c == prev:
            count = count + 1
            if count == 2:
                double_cards.append(prev)
                prev = None
                count = 0
        else:
            prev = c
            count = 1
    mm: [CardCombination] = []
    for i in range(3, int(cards.__len__() / 2) + 1):
        mm += get_straight_with_long(double_cards, i)
    for com in mm:
        get: [Card] = []
        for k in com.Cards():
            i = cards.index(k)
            get.append(k)
            assert(cards[i+1] == k)
            get.append(cards[i+1])
        v = CardCombination(*get)
        ret.append(v)
    return ret


def get_multiThree(cards: [Card]) -> [CardCombination]:
    ret = []
    double_cards: [Card] = []
    prev: Card = None
    count: int = 0
    for c in cards:
        if prev is not None and c == prev:
            count = count + 1
            if count == 3:
                double_cards.append(prev)
                prev = None
                count = 0
        else:
            prev = c
            count = 1
    mm: [CardCombination] = []
    for i in range(2, int(cards.__len__() / 3) + 1):
        mm += get_straight_with_long(double_cards, i)
    for com in mm:
        get: [Card] = []
        for k in com.Cards():
            i = cards.index(k)
            get.append(k)
            assert(cards[i+1] == k)
            assert(cards[i+2] == k)
            get.append(cards[i+1])
            get.append(cards[i+2])
        ret.append(CardCombination(*get))
    return ret


def n_single(cards: [Card], n: int) -> [[Card]]:
    ret = []
    if cards.__len__() < n:
        return ret
    if cards.__len__() == n:
        return [cards.copy()]
    if n == 1:
        return list(map(lambda x: [x], cards))
    a = cards[0]
    k = n_single(cards[1:], n - 1)
    for v in k:
        v.append(a)
    p = n_single(cards[1:], n)
    return k + p


# BUG
def n_double(cards: [Card], n: int) -> [[Card]]:
    ret = []
    dcards = []
    prev: Card = None
    for c in cards:
        if prev is not None and prev == c:
            dcards.append(prev)
            prev = None
        else:
            prev = c
    n = n_single(dcards, n)
    for v in n:
        k = []
        for u in v:
            i = cards.index(u)
            k.append(u)
            assert(cards.__len__() > i + 1)
            k.append(cards[i + 1])
        ret.append(k)
    return ret


# FIXME when REMAIN also in three
def get_multiThreeA(cards: [Card]) -> [CardCombination]:
    ret = []
    mm = get_multiThree(cards)
    for cs in mm:
        c_len = cs.Cards().__len__()
        assert(c_len % 3 == 0)
        s_len = int(c_len / 3)
        n = n_single([c for c in cards if c not in cs.Cards()], s_len)
        for v in n:
            k = cs.copy()
            k.AddCard(*v)
            ret.append(k)
    return ret


# FIXME when REMAIN also is in three
def get_multiThreeB(cards: [Card]) -> [CardCombination]:
    ret = []
    mm = get_multiThree(cards)
    for cs in mm:
        c_len = cs.Cards().__len__()
        assert(c_len % 3 == 0)
        s_len = int(c_len / 3)
        n = n_double([c for c in cards if c not in cs.Cards()], s_len)
        for v in n:
            ret.append(cs.copy().AddCard(*v))
    return ret


def all_valid(kk: [CardCombination]):
    for k in kk:
        assert(k is not None)
        assert(k.valid())


class PlayerState():
    def __init__(self, cards=[], who_lord=0,
                 prev=[], me=[], next=[], who_take=0):
        self.who_lord: int = who_lord
        self.cards: [Card] = cards
        self.prev: [CardCombination] = prev
        self.me: [CardCombination] = []
        self.next: [CardCombination] = next
        self.who_take: int = who_take

    def basic_input_from_state(self) -> [int]:
        data = []
        data.append(self.who_lord)
        data.append(self.who_take)

        def card_to_val(card) -> float:
            return card.card_value + card.card_suit.value * 0.1
        for c in self.cards:
            data.append(card_to_val(c))
        for _ in range(data.__len__(), 22):
            data.append(0)
        for com in self.prev:
            if com is None:
                continue
            for c in com.Cards():
                data.append(card_to_val(c))
        for _ in range(data.__len__(), 42):
            data.append(0)
        for com in self.me:
            if com is None:
                continue
            for c in com.cards():
                data.append(card_to_val(c))
        for _ in range(data.__len__(), 62):
            data.append(0)
        for com in self.next:
            if com is None:
                continue
            for c in com.Cards():
                data.append(card_to_val(c))
        for _ in range(data.__len__(), 82):
            data.append(0)
        assert(data.__len__() == 82)
        return data


class Player():
    def __init__(self, game, player_agent: LordAgent, verbose: int,
                 immediate_train: bool, post_train: bool,
                 random_generate: bool):
        self.__myname = "lord game player"
        self.__verbose = verbose
        self.__game = game
        self.__agent = player_agent
        self.__immediate_train = immediate_train
        self.__post_train = post_train
        self.__random_generate = random_generate
        self.__remember: [([int], int, float)] = []
        self.old_state = None
        self.prev_action = None
        self.prev_quality = None
        self.reward = None
        self.Reset()

    def AssignName(self, name: str):
        self.__myname = name

    def name(self) -> str:
        return self.__myname

    @property
    def is_lord(self) -> bool:
        return self.who_lord == 0

    def finish_this_round_and_train(self, is_lord_win: bool):
        if not self.__post_train or self.__remember.__len__() == 0:
            return
        win = False
        if self.is_lord:
            win = is_lord_win
        else:
            win = not is_lord_win
        state_action = []
        quality = []
        for s, a, q in self.__remember:
            assert(s.__len__() == 82)
            if win:
                q = q + 2
            else:
                q = q - 2
            state_action.append((s, a))
            quality.append(q)
        self.__game.save_train_data(state_action, quality)
        self.__remember = []

    def Reset(self):
        self.holded_cards = []
        self.prev_player_history = []
        self.my_history = []
        self.next_player_history = []
        self.who_lord = - 2
        self.who_take = -2

    def GetState(self) -> [int]:
        return PlayerState(cards=self.holded_cards, who_lord=self.who_lord,
                           prev=self.prev_player_history, me=self.my_history,
                           next=self.next_player_history,
                           who_take=self.who_take).basic_input_from_state()

    def GetCards(self, cards: [Card]):
        self.holded_cards = self.holded_cards + cards
        self.holded_cards.sort(key=card_sort)

    def GetLordCards(self, cards: [Card]):
        self.GetCards(cards)
        self.who_take = 0
        self.who_lord = 0

    def __allPossibleCombination(self) -> [CardCombination]:
        single = get_single(self.holded_cards)
        double = get_double(self.holded_cards)
        three = get_three(self.holded_cards)
        threeA = get_threeA(self.holded_cards)
        threeB = get_threeB(self.holded_cards)
        four = get_four(self.holded_cards)
        fourandtwo = get_fourAndTwo(self.holded_cards)
        straight = get_straight(self.holded_cards)
        multiDouble = get_multiDouble(self.holded_cards)
        multiThree = get_multiThree(self.holded_cards)
        multiThreeA = get_multiThreeA(self.holded_cards)
        multiThreeB = get_multiThreeB(self.holded_cards)
        all_valid(single)
        all_valid(double)
        all_valid(three)
        all_valid(threeA)
        all_valid(threeB)
        all_valid(four)
        all_valid(fourandtwo)
        all_valid(straight)
        all_valid(multiDouble)
        all_valid(multiThree)
        all_valid(multiThreeA)
        all_valid(multiThreeB)
        return (single + double + three + threeA + threeB + four +
                fourandtwo + straight + multiDouble +
                multiThree + multiThreeA + multiThreeB)

    def doit(self) -> CardCombination:
        assert(self.who_take >= -1 and self.who_take <= 1)
        prevComb = None
        if self.who_take == -1:
            prevComb = self.prev_player_history[-1]
        elif self.who_take == 1:
            prevComb = self.next_player_history[-1]
        all_comb = self.__allPossibleCombination()
        available = list(
            filter(
                lambda x: (prevComb is None or
                           x.GreaterThan(prevComb)),
                all_comb)
        )
        al = available.__len__()
        the_comb: CardCombination = None
        current_state = self.GetState()
        quality: float = 0
        action: int = 0
        if self.__random_generate:
            action = random.randint(0, al)
            quality = self.__agent.quality(current_state, action)
        else:
            quality, action = self.__agent.predict(current_state, al)
        assert(action >= 0 and action <= al)
        self.prev_quality = quality
        self.prev_action = action
        self.old_state = current_state
        self.reward = 0
        if self.__post_train:
            self.__remember.append((current_state, action, quality))
        if action == 0 and self.who_take == 0:
            assert(not self.over())
            self.reward = self.reward - 5
            action = 1
        if action > 0:
            the_comb = available[action - 1]
        if the_comb is not None:
            if self.__verbose >= 2:
                print(self.__myname, "-> ", the_comb)
            for i in the_comb.Cards():
                self.holded_cards.remove(i)
        else:
            self.reward = self.reward - 2
        return the_comb

    def inform(self, cards: CardCombination, who: int):
        assert(who >= -1 and who <= 1)
        if who == -1:
            self.prev_player_history.append(cards)
        elif who == 0:
            self.my_history.append(cards)
        else:
            self.next_player_history.append(cards)
        if cards is not None:
            self.who_take = who
        if who == 0 and self.__immediate_train:
            assert(self.old_state is not None)
            assert(self.prev_quality is not None)
            assert(self.prev_action is not None)
            if self.over():
                self.reward = 100
            else:
                self.reward = self.reward - 1
            self.__agent.train_q(self.old_state, self.prev_action,
                                 self.prev_quality, self.reward,
                                 self.GetState())

    def ShowCards(self):
        the_str = "Farmer " + self.__myname + ": "
        if self.who_lord == 0:
            the_str = "$Lord$ " + self.__myname + ": "
        for i in reversed(self.holded_cards):
            the_str = the_str + str(i)
        if self.__verbose >= 2:
            print(the_str)

    def over(self) -> bool:
        return self.holded_cards.__len__() == 0


class LordGame():
    def __init__(self, verbose: int = 0, random_action: bool = False,
                 immediate_train: bool = True, post_train: bool = True,
                 gameround: int = 1,
                 model_path: str = "./lordgamemodel"):
        self.player_agent = LordAgent()
        self.__verbose = verbose
        self.__model_path = model_path
        self.__game_counter = 0
        self.__round = gameround
        self.__round_counter = 1
        self.__random_generate = True
        self.player_agent.try_load(self.__model_path)
        self.p1 = Player(self, self.player_agent, verbose, immediate_train,
                         post_train, random_action)
        self.p2 = Player(self, self.player_agent, verbose, immediate_train,
                         post_train, random_action)
        self.p3 = Player(self, self.player_agent, verbose, immediate_train,
                         post_train, random_action)
        self.__train = immediate_train or post_train
        self.__train_data_x = []
        self.__train_data_y = []
        self.p1.AssignName("p1")
        self.p2.AssignName("p2")
        self.p3.AssignName("p3")
        self.who_next: int = -1
        self.all_cards: [Card] = []
        for i in range(3, 16):
            self.all_cards.append(Card(i, CardSuit.ClubSuit))
            self.all_cards.append(Card(i, CardSuit.DiamondSuit))
            self.all_cards.append(Card(i, CardSuit.HeartSuit))
            self.all_cards.append(Card(i, CardSuit.SpadeSuit))
        self.all_cards.append(Card(16, CardSuit.HeartSuit))
        self.all_cards.append(Card(16, CardSuit.SpadeSuit))

    def save_train_data(self, x, y):
        assert(x.__len__() == y.__len__())
        for x_v in x:
            self.__train_data_x.append(x_v)
        for y_v in y:
            self.__train_data_y.append(y_v)

    def game_round(self):
        while not self.GameOver():
            assert(self.who_next >= 0 and self.who_next <= 2)
            next_cards: CardCombination = None
            if self.who_next == 0:
                next_cards = self.p1.doit()
            elif self.who_next == 1:
                next_cards = self.p2.doit()
            elif self.who_next == 2:
                next_cards = self.p3.doit()

            def who_reg(a, b):
                k = a - b
                if k == 2:
                    return -1
                elif k == -2:
                    return 1
                return k
            self.p1.inform(next_cards, who_reg(self.who_next, 0))
            self.p2.inform(next_cards, who_reg(self.who_next, 1))
            self.p3.inform(next_cards, who_reg(self.who_next, 2))
            self.who_next = (self.who_next + 1) % 3
        lord_win = False
        if((self.p1.over() and self.p1.is_lord) or
           (self.p2.over() and self.p2.is_lord) or
           (self.p3.over() and self.p3.is_lord)):
            lord_win = True
        self.p1.finish_this_round_and_train(lord_win)
        self.p2.finish_this_round_and_train(lord_win)
        self.p3.finish_this_round_and_train(lord_win)
        if self.__game_counter % 10 == 0 and self.__train:
            if self.__train_data_x.__len__() > 0:
                self.player_agent.train_x(self.__train_data_x,
                                          self.__train_data_y)
                self.__train_data_x = []
                self.__train_data_y = []
            self.player_agent.save(self.__model_path)
        self.__game_counter = self.__game_counter + 1
        who_over = None
        win2 = None
        if self.p1.over():
            who_over = self.p1
        elif self.p2.over():
            who_over = self.p2
        else:
            who_over = self.p3
        if not who_over.is_lord:
            if not self.p1.is_lord and self.p1 != who_over:
                win2 = self.p1
            if not self.p2.is_lord and self.p2 != who_over:
                win2 = self.p2
            if not self.p3.is_lord and self.p3 != who_over:
                win2 = self.p3
        msg = who_over.name()
        if win2 is not None:
            msg = msg + " and " + win2.name()
        msg += " win"
        if who_over.is_lord:
            msg += "s (Lord) "
        else:
            msg += " (Farmer) "
        msg += "round "
        msg += str(self.__round_counter)
        self.__round_counter = self.__round_counter + 1
        if self.__verbose >= 1:
            print(msg)

    def __run_game(self):
        new_cards = self.all_cards.copy()
        rand_cards = []
        while new_cards.__len__() > 0:
            r = random.randint(0, new_cards.__len__() - 1)
            rand_cards.append(new_cards[r])
            del new_cards[r]
        all_len = rand_cards.__len__()
        assert(all_len == self.all_cards.__len__())
        assert(new_cards.__len__() == 0)
        n = all_len - 3
        assert(n % 3 == 0)
        assert(self.p1.over())
        assert(self.p2.over())
        assert(self.p3.over())
        self.p1.GetCards(rand_cards[0:int(n / 3)])
        self.p2.GetCards(rand_cards[int(n / 3):int(n * 2 / 3)])
        self.p3.GetCards(rand_cards[int(n * 2 / 3):n])
        del rand_cards[0:n]
        assert(rand_cards.__len__() == 3)
        self.who_next = random.randint(0, 2)
        # TODO 叫地主
        if self.who_next == 0:
            self.p1.GetLordCards(rand_cards)
            self.p2.who_lord = -1
            self.p3.who_lord = 1
        elif self.who_next == 1:
            self.p1.who_lord = 1
            self.p2.GetLordCards(rand_cards)
            self.p3.who_lord = -1
        elif self.who_next == 2:
            self.p1.who_lord = -1
            self.p2.who_lord = 1
            self.p3.GetLordCards(rand_cards)
        self.p1.ShowCards()
        self.p2.ShowCards()
        self.p3.ShowCards()
        try:
            self.game_round()
        except AssertionError as err:
            print(err)
        finally:
            self.p1.Reset()
            self.p2.Reset()
            self.p3.Reset()

    def RunAGame(self):
        while self.__round > 0:
            self.__run_game()
            self.__round = self.__round - 1

    def GameOver(self) -> bool:
        return self.p1.over() or self.p2.over() or self.p3.over()


# train first model using random generate action
#     (random_action=True and post_train=True)
# optimize the model using Q-learning algorithm, very slow
#     (random_action=False immediate_train=True)
if __name__ == "__main__":
    game = LordGame(2, random_action=False, immediate_train=False,
                    post_train=False, gameround=100)
    game.RunAGame()
