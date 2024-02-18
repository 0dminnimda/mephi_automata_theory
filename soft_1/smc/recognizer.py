from collections import deque
from recognizer_sm import Recognizer_sm, MainMap


class Recognizer:
    def __init__(self, string: str):
        self.string = string
        self.position = 0
        self.finished = False
        self.saved = deque()
        self._fsm = Recognizer_sm(self)

    def peek(self, at: int = 0):
        pos = self.position + at
        if pos >= len(self.string):
            return ""
        return self.string[pos]

    def save_position(self):
        self.saved.append(self.position)

    def consume(self, amount: int = 1):
        self.position += amount
        self.position = min(self.position, len(self.string))

    def match(self, string: str):
        if self.string[self.position : self.position + len(string)] == string:
            return True
        return False

    def match_and_consume(self, string: str):
        if self.match(string):
            self.position += len(string)
            return True
        return False

    def finish(self):
        self.finished = True

    def get_captures(self):
        for _ in range(0, len(self.saved), 2):
            start = self.saved.popleft()
            end = self.saved.popleft()
            yield self.string[start:end]

    def next(self):
        return self._fsm.next()

    def getState(self):
        return self._fsm.getState()


def has_duplicates(x):
    seen = set()
    return any(i in seen or seen.add(i) for i in x)


def match(string):  # type: (str) -> tuple[bool, str | None]
    rec = Recognizer(string)
    while not rec.finished:
        # print(rec.peek(), rec.getState().getName())
        rec.next()

    if rec.getState() is not MainMap.match:
        return False, None

    # assert len(rec.saved) % 2 == 0
    captures = rec.get_captures()
    name = next(captures)
    if has_duplicates(captures):
        return True, name
    return True, None
