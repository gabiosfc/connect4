import numpy as np
import pygame
import sys
import math
import random
import time
import threading

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

minimax_total_time = 0  # Tempo total para o algoritmo Minimax
alpha_beta_total_time = 0  # Tempo total para o algoritmo AlphaBeta


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
    global minimax_total_time  # Usar a variável global de tempo do Minimax
    start_time = time.time()  # Marcar o início do tempo
    
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
        end_time = time.time()  # Marcar o fim do tempo
        minimax_total_time += (end_time - start_time)  # Acumular o tempo de execução
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
        end_time = time.time()  # Marcar o fim do tempo
        minimax_total_time += (end_time - start_time)  # Acumular o tempo de execução
        return column, value


def minimax_alpha_beta(board, depth, alpha, beta, maximizingPlayer):
    global alpha_beta_total_time  # Usar a variável global de tempo do AlphaBeta
    start_time = time.time()  # Marcar o início do tempo
    
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
        end_time = time.time()  # Marcar o fim do tempo
        alpha_beta_total_time += (end_time - start_time)  # Acumular o tempo de execução
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
        end_time = time.time()  # Marcar o fim do tempo
        alpha_beta_total_time += (end_time - start_time)  # Acumular o tempo de execução
        return column, value


def select_depth():
    screen.fill(WHITE)
    font = pygame.font.SysFont("monospace", 25)
    
    depth_rects = [pygame.Rect(100 + (i - 1) * 60, 400, 50, 50) for i in range(1, 7)]    
    
    for rect in depth_rects:
        pygame.draw.rect(screen, GRAY, rect)
    
    text_depth = font.render("Profundidade", 1, BLACK)
    screen.blit(text_depth, (100, 330))

    # Números de profundidade
    for i in range(1, 7):
        depth_text = font.render(str(i), 1, BLACK)
        screen.blit(depth_text, (depth_rects[i - 1].x + (depth_rects[i - 1].width - depth_text.get_width()) // 2, depth_rects[i - 1].y + (depth_rects[i - 1].height - depth_text.get_height()) // 2))

    pygame.display.update()

    selected_depth = None 

    while selected_depth is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Verificando profundidade ao clicar
            if event.type == pygame.MOUSEBUTTONDOWN:
                for i in range(1, 7):
                    if depth_rects[i - 1].collidepoint(event.pos):
                        selected_depth = i
                        print(f"Profundidade selecionada: {selected_depth}")
                        break  # Sai do loop após selecionar uma profundidade

    # Exibir seleção do usuário
    screen.fill(WHITE)
    selected_depth_text = font.render(f"Profundidade: {selected_depth}", 1, BLACK)
    screen.blit(selected_depth_text, (width // 2 - selected_depth_text.get_width() // 2, height // 2))
    pygame.display.update()

    pygame.time.wait(2000)

    return selected_depth

# Função para o algoritmo Minimax
def run_minimax(board, depth, turn):
    global minimax_total_time
    start_time = time.time()
    col, minimax_score = minimax(board, depth, True)
    end_time = time.time()
    minimax_total_time += (end_time - start_time)
    return col

# Função para o algoritmo Alpha-Beta
def run_alpha_beta(board, depth, turn):
    global alpha_beta_total_time
    start_time = time.time()
    col, alpha_beta_score = minimax_alpha_beta(board, depth, -math.inf, math.inf, True)
    end_time = time.time()
    alpha_beta_total_time += (end_time - start_time)
    return col

def main():
    board = create_board()
    game_over = False
    turn = random.randint(PLAYER, AI)

    # Chama a função de seleção de profundidade
    depth = select_depth()

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
            # Criando threads para rodar ambos os algoritmos ao mesmo tempo
            minimax_thread = threading.Thread(target=run_minimax, args=(board, depth, turn))
            alpha_beta_thread = threading.Thread(target=run_alpha_beta, args=(board, depth, turn))

            # Iniciar ambos os threads
            minimax_thread.start()
            alpha_beta_thread.start()

            # Esperar ambos os threads terminarem
            minimax_thread.join()
            alpha_beta_thread.join()


            algorithm_choice = random.choice(["Minimax", "Alpha-Beta"])  
            print(f"Algoritmo escolhido: {algorithm_choice}")  # Exibe o algoritmo escolhido no console
            
            if algorithm_choice == "Minimax":
                col = run_minimax(board, depth, turn)
            else:
                col = run_alpha_beta(board, depth, turn)

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

    # Exibir o tempo total de execução de cada algoritmo ao final
    print(f"Tempo total de execução do Minimax: {minimax_total_time:.4f} segundos")
    print(f"Tempo total de execução do AlphaBeta: {alpha_beta_total_time:.4f} segundos")

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
