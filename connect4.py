import numpy as np
import pygame
import sys
import math
import random

# Configurações básicas
ROW_COUNT = 7
COLUMN_COUNT = 8
SQUARESIZE = 80
RADIUS = int(SQUARESIZE / 2 - 8)
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE
size = (width, height)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (192, 192, 192)
BLACK = (0, 0, 0)

# Posições das peças
PLAYER = 0
AI = 1
PLAYER_PIECE = 1
AI_PIECE = 2
WINDOW_LENGTH = 4
EMPTY = 0

# Inicialização do Pygame
pygame.init()
screen = pygame.display.set_mode(size)
font = pygame.font.SysFont("monospace", 40)

def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board

def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def winning_move(board, piece):
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][c + 3] == piece:
                return True
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][c] == piece:
                return True
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece:
                return True
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece:
                return True
    return False

def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    # Pontuação para quatro peças seguidas
    if window.count(piece) == 4:
        score += 100
    # Pontuação para três peças seguidas com um vazio
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    # Pontuação para duas peças seguidas com dois vazios
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2
    # Penalização para três peças do oponente com um vazio
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4
    # Penalização para duas peças do oponente com dois vazios
    elif window.count(opp_piece) == 2 and window.count(EMPTY) == 2:
        score -= 1
    return score

def score_position(board, piece):
    score = 0

    # Avaliação do centro
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Avaliação das linhas
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Avaliação das colunas
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Avaliação das diagonais principais
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Avaliação das diagonais secundárias
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score


def get_valid_locations(board):
    valid_locations = [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]
    return valid_locations

def minimax(board, depth, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(valid_locations) == 0
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, AI_PIECE))

    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth - 1, False)[1]
            if new_score > value:
                value = new_score
                column = col
        return column, value
    else:
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth - 1, True)[1]
            if new_score < value:
                value = new_score
                column = col
        return column, value

def minimax_alpha_beta(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(valid_locations) == 0
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, AI_PIECE))

    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax_alpha_beta(b_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else:
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax_alpha_beta(b_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

def select_algorithm_and_depth():
    screen.fill(WHITE)
    font = pygame.font.SysFont("monospace", 25)
    
    # Definindo áreas dos botões
    depth_rects = [pygame.Rect(100 + (i - 1) * 60, 400, 50, 50) for i in range(1, 7)]
    minimax_rect = pygame.Rect(100, 150, 250, 60)
    alpha_beta_rect = pygame.Rect(100, 230, 250, 60)
    
    
    # Definindo as cores dos botões
    for rect in depth_rects:
        pygame.draw.rect(screen, GRAY, rect)
    pygame.draw.rect(screen, GRAY, minimax_rect)
    pygame.draw.rect(screen, GRAY, alpha_beta_rect)
    
    
    # Texto nos botões
    text_depth = font.render("Profundidade", 1, BLACK)
    text_minimax = font.render("Minimax", 1, BLACK)
    text_alpha_beta = font.render("AlphaBeta", 1, BLACK)
    
    
    # Centralizando o texto nos botões
    screen.blit(text_minimax, (minimax_rect.x + (minimax_rect.width - text_minimax.get_width()) // 2, minimax_rect.y + (minimax_rect.height - text_minimax.get_height()) // 2))
    screen.blit(text_alpha_beta, (alpha_beta_rect.x + (alpha_beta_rect.width - text_alpha_beta.get_width()) // 2, alpha_beta_rect.y + (alpha_beta_rect.height - text_alpha_beta.get_height()) // 2))
    screen.blit(text_depth, (100, 330))

    # Números de profundidade
    for i in range(1, 7):
        depth_text = font.render(str(i), 1, BLACK)
        screen.blit(depth_text, (depth_rects[i - 1].x + (depth_rects[i - 1].width - depth_text.get_width()) // 2, depth_rects[i - 1].y + (depth_rects[i - 1].height - depth_text.get_height()) // 2))

    pygame.display.update()

    selected_algorithm = None
    selected_depth = None 

    while selected_algorithm is None or selected_depth is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if minimax_rect.collidepoint(event.pos):
                    selected_algorithm = "minimax"
                elif alpha_beta_rect.collidepoint(event.pos):
                    selected_algorithm = "alpha_beta"
                
                # Verificando profundidade
                for i in range(1, 7):
                    if depth_rects[i - 1].collidepoint(event.pos):
                        selected_depth = i

    # Exibir seleção do usuário
    screen.fill(WHITE)
    selected_text = font.render(f"Algoritmo: {selected_algorithm.capitalize()}", 1, BLACK)
    selected_depth_text = font.render(f"Profundidade: {selected_depth}", 1, BLACK)
    screen.blit(selected_text, (width // 2 - selected_text.get_width() // 2, height // 2 - 60))
    screen.blit(selected_depth_text, (width // 2 - selected_depth_text.get_width() // 2, height // 2))
    pygame.display.update()

    pygame.time.wait(2000)

    return selected_algorithm, selected_depth

def main():
    board = create_board()
    game_over = False
    turn = random.randint(PLAYER, AI)

    # Seleção de algoritmo e profundidade
    selected_algorithm, selected_depth = select_algorithm_and_depth()

    # Loop do jogo
    draw_board(board)
    pygame.display.update()

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, WHITE, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)
            pygame.display.update()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, WHITE, (0, 0, width, SQUARESIZE))
                if turn == PLAYER:
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))
                    if is_valid_location(board, col):
                        row = get_next_open_row(board, col)
                        drop_piece(board, row, col, PLAYER_PIECE)
                        if winning_move(board, PLAYER_PIECE):
                            label = font.render("Você Venceu!", 1, RED)
                            screen.blit(label, (40, 10))
                            game_over = True
                        turn += 1
                        turn = turn % 2
                        draw_board(board)

        if turn == AI and not game_over:
            if selected_algorithm == "minimax":
                col, minimax_score = minimax(board, selected_depth, True)
            elif selected_algorithm == "alpha_beta":
                col, minimax_score = minimax_alpha_beta(board, selected_depth, -math.inf, math.inf, True)
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)
                if winning_move(board, AI_PIECE):
                    label = font.render("IA Venceu!", 1, BLUE)
                    screen.blit(label, (40, 10))
                    game_over = True
                draw_board(board)
                turn += 1
                turn = turn % 2
        if game_over:
            pygame.time.wait(3000)

def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, GRAY, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, WHITE, (int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, BLUE, (int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()

if __name__ == "__main__":
    main()
