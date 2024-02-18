from recognizer_sm import Recognizer_sm, MainMap


class Recognizer:
    def __init__(self, string: str):
        self.string = string
        self.position = 0
        self.finished = False
        self._fsm = Recognizer_sm(self)

    def peek(self, at: int = 0):
        pos = self.position + at
        if pos >= len(self.string):
            return ""
        return self.string[pos]

    def consume(self):
        if self.position < len(self.string):
            self.position += 1

    def match_and_consume(self, string: str):
        if self.string[self.position : self.position + len(string)] == string:
            self.position += len(string)
            return True
        return False        

    def finish(self):
        self.finished = True

    def next(self):
        return self._fsm.next()

    def getState(self):
        return self._fsm.getState()


try:
    s = input()
except EOFError:
    s = ""

rec = Recognizer(s)
while not rec.finished:
    # print(rec.peek(), rec.getState().getName())
    rec.next()


exit(rec.getState() is not MainMap.match)
