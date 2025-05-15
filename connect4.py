ROWS, COLS = 6, 7

def create_board():
    return [[' ' for _ in range(COLS)] for _ in range(ROWS)]

def print_board(board):
    for row in board:
        print('| ' + ' | '.join(row) + ' |')
    print('+---' * COLS + '+')
    print('  ' + '   '.join(map(str, range(COLS))))

def get_valid_moves(board):
    return [c for c in range(COLS) if board[0][c] == ' ']

def drop_disc(board, col, disc):
    # Check if the column is already full (top cell is not empty)
    if board[0][col] != ' ':
        return None, False # Return False for success if the column is full from the top

    # Find the lowest empty row in the column
    for row in reversed(range(ROWS)):
        if board[row][col] == ' ':
            board[row][col] = disc
            return row, True # Return True for success when a disc is dropped

    # This part should ideally not be reached if the top check is done first,
    # but as a safeguard, return False if no empty row was found
    return None, False # Return False for success if no empty row found

def is_winning_move(board, row, col, disc, return_type=False):
    def count_discs(delta_row, delta_col):
        count = 0
        r, c = row + delta_row, col + delta_col
        while 0 <= r < len(board) and 0 <= c < len(board[0]) and board[r][c] == disc:
            count += 1
            r += delta_row
            c += delta_col
        return count

    directions = {
        "horizontal": [(0, -1), (0, 1)],
        "vertical": [(-1, 0), (1, 0)],
        "diag_up": [(-1, -1), (1, 1)],
        "diag_down": [(-1, 1), (1, -1)]
    }

    for name, (dir1, dir2) in directions.items():
        count = 1 + count_discs(*dir1) + count_discs(*dir2)
        if count >= 4:
            if return_type:
                return name  # Return "horizontal", "vertical", etc
            else:
                return True

    if return_type:
        return None
    else:
        return False

def is_draw(board):
    return all(cell != ' ' for cell in board[0])
