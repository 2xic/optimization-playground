
class TokenTracker:
    def __init__(self) -> None:
        self.token_index = {}
        self.index_token = {}

    def add_token(self, token):
        if token not in self.token_index:
            index = len(self.token_index)
            self.token_index[token] = index
            self.index_token[index] = token
            return index
        else:
            return self.token_index[token]
        
class SimpleTokenizer:
    def __init__(self) -> None:
        self.tokens_tracker = TokenTracker()
        self._PADDING = self.tokens_tracker.add_token("<PADDING>")
        self.start_offset = self.tokens_tracker.add_token("<START_OFFSET>")
        self.end_offset = self.tokens_tracker.add_token("<END_OFFSET>")
        self.start_instruction = self.tokens_tracker.add_token("<START_INSTRUCTION>")
        self.end_instruction = self.tokens_tracker.add_token("<END_T_INSTRUCTION>")
