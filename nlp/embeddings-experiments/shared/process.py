
class Process:
    def __init__(self) -> None:
        pass

    def process(self, text):
        return self.clean_tokens(
            text.lower().split(" ")
        )
    
    def clean_tokens(self, tokens):
        output = []
        for i in tokens:
            token = i
            for j in [self.remove_links]:
                out = j(i)
                if out is None:
                    token = None
                    break
            if token is not None:
                output.append(token)
        return output

    def remove_links(self, token):
        if "http" in token:
            return None
        return token
    