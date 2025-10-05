from collections import deque
class RabbitLeap:
    def __init__(self):
        self.start = "EEE_WWW"
        self.goal = "WWW_EEE"

    def get_successors(self, state):
        # get all valid next states
        moves = []
        blank = state.index('_')

        # east rabbits move or jump right
        if blank > 0 and state[blank - 1] == 'E':
            moves.append((blank - 1, blank))
        if blank > 1 and state[blank - 1] == 'W' and state[blank - 2] == 'E':
            moves.append((blank - 2, blank))

        # west rabbits move or jump left
        if blank < 6 and state[blank + 1] == 'W':
            moves.append((blank + 1, blank))
        if blank < 5 and state[blank + 1] == 'E' and state[blank + 2] == 'W':
            moves.append((blank + 2, blank))

        successors = []
        for frm, to in moves:
            new_state = list(state)
            new_state[frm], new_state[to] = new_state[to], new_state[frm]
            successors.append(''.join(new_state))
        
        return successors

    def bfs(self):
        # bfs implementation
        q = deque([(self.start, [])])
        visited = {self.start}
        nodes_explored = 0

        while q:
            curr, path = q.popleft()
            nodes_explored += 1

            if curr == self.goal:
                print("bfs solution:")
                print(f"step cnt: {len(path)}")
                for state in path + [curr]:
                    print(' '.join(state))
                return path + [curr], nodes_explored

            for nxt in self.get_successors(curr):
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, path + [curr]))

        return None, nodes_explored

    def dfs(self):
        # dfs implementation
        stack = [(self.start, [])]
        visited = set()
        nodes_explored = 0

        while stack:
            curr, path = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            nodes_explored += 1

            if curr == self.goal:
                print("\ndfs solution:")
                print(f"step cnt: {len(path)}")
                for state in path + [curr]:
                    print(' '.join(state))
                return path + [curr], nodes_explored

            for nxt in reversed(self.get_successors(curr)):
                if nxt not in visited:
                    stack.append((nxt, path + [curr]))

        return None, nodes_explored

def main():
    game = RabbitLeap()
    bfs_result, bfs_nodes = game.bfs()
    dfs_result, dfs_nodes = game.dfs()

if __name__ == "__main__":
    main()
