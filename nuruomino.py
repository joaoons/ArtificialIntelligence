# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 95:
# ist1103243 João Santos
# Diogo Ceia

import sys
import numpy as np
from collections import defaultdict
from search import Problem
from search import depth_first_graph_search
from search import breadth_first_graph_search
from search import depth_first_tree_search 
from search import breadth_first_tree_search
from collections import deque


class Tetromino:
    def __init__(self, label, base_shape):
        self.label = label
        self.base_shape = np.array(base_shape)
        self._variants = self._generate_variants()

    def _generate_variants(self):
        variants = set()

        for k in range(4):
            # roda a matriz do tetronimo 90 x 4
            rotated = np.rot90(self.base_shape, k)

            # espelha a matriz resultante em relaco ao eixo x
            # percorreno assim todas as matrizes variantes de um tetronimo
            for flip in [rotated, np.fliplr(rotated)]:
                key = tuple(tuple(row) for row in flip.tolist())
                variants.add(key)
        return [np.array(v) for v in variants]

    def variants(self):
        return self._variants

    def __repr__(self):
        return f"Tetromino('{self.label}')"

class Board:
    def __init__(self, matrix):
        self.matrix = matrix
        self.rows = len(matrix)
        self.cols = len(matrix[0]) if matrix else 0
        # dicionario que define as regioes da board
        # {region_id:[cell(x,y)]}
        # ex: { 1: [(1,1), (1,2), (1,3), (2,1)]}
        self.regions = self._compute_regions()

    def _compute_regions(self):
        regions = defaultdict(list)
        for r in range(self.rows):
            for c in range(self.cols):
                region_id = self.matrix[r][c]
                if not isinstance(region_id, str) and region_id is not None:
                    regions[region_id].append((r, c))

        # ordena as regioes da menor para a maior
        return regions

    def get_value(self, row, col):
        return self.matrix[row][col]

    def print_instance(self):
        for row in self.matrix:
            print("\t".join(str(cell) for cell in row))
        # le o ficheiro input linha a linha e transforma na matriz correspondente
        #

    # devolve a board com a respetiva matriz
    @staticmethod
    def parse_instance():
        matrix = []
        for line in sys.stdin:
            if line.strip() == "":
                continue
            matrix.append([int(x) for x in line.strip().split()])
        return Board(matrix)

    def key(self):
        return tuple(tuple(row) for row in self.matrix)

    # procura zonas adjacentes ortogonais
    def adjacent_regions(self, region_id):
        adjacents = set()

        # cima, baixo, esquerda, direita
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # percorre a matriz posicao a posicao
        # adicona os deltas direcao a cada posicao
        # se a nova posicao pertentece a um region que nao seja
        # a regiao atual acrescente o id as regioes adjacentes
        for x, y in self.regions[region_id]:
            for dx, dy in directions:
                nx = x + dx
                ny = y + dy

                # verifica se novas coordenadas estao dentro da matriz
                if 0 <= nx < self.rows and 0 <= ny < self.cols:
                    neighbor_id = self.matrix[nx][ny]
                    if (
                        neighbor_id != region_id
                        and neighbor_id is not None
                        and isinstance(neighbor_id, int)
                    ):
                        adjacents.add(neighbor_id)
        return list(adjacents)

 
    # usa o mesmo conceito do adjacent_regions
    # mas desta vez nao e necessario percorrer a matriz
    # dada a cordenada e so calcular as posicoes
    # a volta dessa cordenada
    def adjacent_values(self, row, col):
        values = []
        for dx, dy in self.orthogonal_directions():
            nx = row + dx
            ny = col + dy

            # verifica se novas coordenadas estao dentro da matriz
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                values.append(self.matrix[nx][ny])
        return values

    def is_piece_cell(self, r, c):
        val = self.matrix[r][c]
        return isinstance(val, str) and val is not None

    def region_fits_tetronimo(self, region_id, piece: Tetromino, tetromino_matrix):
        region_cells = self.regions[region_id]

        rows = [r for r, _ in region_cells]
        cols = [c for _, c in region_cells]
        min_r, min_c = min(rows), min(cols)
        max_r, max_c = max(rows), max(cols)

        region_set = {(r - min_r, c - min_c) for r, c in region_cells}
        tr, tc = tetromino_matrix.shape

        if tr > len(rows):
            return []

        if tc > len(cols):
            return []

        valid_positions = []

        for r_off in range(0, max_r - min_r - tr + 2):
            for c_off in range(0, max_c - min_c - tc + 2):
                mapped = []
                is_valid = True

                for i in range(tr):
                    for j in range(tc):
                        if tetromino_matrix[i, j]:
                            cell = (i + r_off, j + c_off)
                            if cell not in region_set:
                                is_valid = False
                                break
                            mapped.append((cell[0] + min_r, cell[1] + min_c))
                    if not is_valid:
                        break
                if not is_valid:
                    continue

                # verificação de peças adjacentes com a mesma label
                for r, c in mapped:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            val = self.matrix[nr][nc]
                            if isinstance(val, str) and val == piece.label:
                                is_valid = False
                                break
                    if not is_valid:
                        break
                if not is_valid:
                    continue

                surrounded = True
                for r, c in mapped:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            neighbor_val = self.matrix[nr][nc]
                            #e qualquer celula adjacente nao for da regiao e naofor parte da peça
                            if (nr, nc) not in mapped and neighbor_val != region_id:
                                surrounded = False
                                break
                    if not surrounded:
                        break
                if surrounded:
                    continue

                # Verificação de blocos 2x2
                for r, c in mapped:
                    for dr in [0, -1]:
                        for dc in [0, -1]:
                            square = [
                                (r + dr, c + dc),
                                (r + dr + 1, c + dc),
                                (r + dr, c + dc + 1),
                                (r + dr + 1, c + dc + 1),
                            ]
                            if all(
                                0 <= x < self.rows and 0 <= y < self.cols
                                for x, y in square
                            ):
                                if all(
                                    isinstance(self.matrix[x][y], str)
                                    or (x, y) in mapped
                                    for x, y in square
                                ):
                                    is_valid = False
                                    break
                        if not is_valid:
                            break
                    if not is_valid:
                        break

                if is_valid:
                    valid_positions.append((r_off + min_r, c_off + min_c))

        return valid_positions

    # coloca a matriz do tetronimo na posicao pos
    def place_piece_at_pos(self, region_id, piece, tetromino_matrix, pos):
        pr, pc = tetromino_matrix.shape
        new_matrix = [row[:] for row in self.matrix]  # cópia leve da matriz

        region_cells = self.regions[region_id]
        for r, c in region_cells:
            new_matrix[r][c] = None

        for i in range(pr):
            for j in range(pc):
                if tetromino_matrix[i, j]:
                    new_matrix[i + pos[0]][j + pos[1]] = piece.label

        return Board(new_matrix)


    #verifica se baord tem regioes isoladas (regiao rodeada de Nones)
    def has_isolated_regions(self):
        for region in self.regions.items():
            if self.is_region_isolated(region):
                # Encontrou uma região isolada
                return True  
        # Nenhuma região está isolada
        return False  

    #verifica se uma regiao da board esta isolada (rodeada de None)
    def is_region_isolated(self, region):
        region_cells =region[1]
        for r, c in region_cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.matrix[nr][nc] is not None and self.matrix[nr][nc] != region:
                        # Pelo menos um vizinho pertence a outra região (nao e isolada)
                        return False  
        return True  


    #verifica se  na board eixstem pecas isoladas (rodeada de None)
    def has_isolated_piece(self):
        visited = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if self.matrix[r][c] is not None and isinstance(self.matrix[r][c], str) and (r, c) not in visited:
                    if self.is_piece_isolated( r, c, visited):
                        return True  # Found an isolated piece
        return False

    #verifica se uma peca esta isolada
    def is_piece_isolated(self, r, c, visited):
        label = self.matrix[r][c]
        piece_cells = []
        stack = [(r, c)]
        visited.add((r, c))

        #calculas celulas ocupadas pela peca
        while stack:
            pr, pc = stack.pop()
            piece_cells.append((pr, pc))
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = pr + dr, pc + dc
                #se esta dentro da matriz
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if (nr, nc) not in visited:
                        if self.matrix[nr][nc] == label:
                            stack.append((nr, nc))
                            visited.add((nr, nc))

        # Verifica se há alguma célula adjacente (a qualquer célula da peça)
        # que seja outra peça ou uma célula de região (um numeor)
        # se existir a peca nao esta isolada
        for (pr, pc) in piece_cells:
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = pr + dr , pc + dc
                #se esta dentro da matriz
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbor = self.matrix[nr][nc]
                    if neighbor is not None and (neighbor != label):
                        return False  # Tem conexão com outra região ou peça
        return True  # Está completamente isolada



    #calcula se a baord apresenta um nurikabe valido (solucao)
    def is_valid_nurikabe(self):
        visited = set()

        # verifica se todas as regioes estao preenchidas (noa a numeros na board) 
        # so Nones e letras
        for r in range(self.rows):
            for c in range(self.cols):
                if isinstance(self.matrix[r][c], int):
                    return False

        # encontrar a primeira célula de peça
        # para inciar o verificao de conetividade
        start = None
        for r in range(self.rows):
            for c in range(self.cols):
                if isinstance(self.matrix[r][c], str):
                    start = (r, c)
                    break
            if start:
                break

        if not start:
            return False  # não há nenhuma peça no tabuleiro

        # BFS para verificar conectividade das peças
        stack = [start]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    val = self.matrix[nr][nc]
                    if isinstance(val, str) and (nr, nc) not in visited:
                        stack.append((nr, nc))

        # verificar se todas as células com letras foram visitadas
        total_pieces = sum(
            1
            for r in range(self.rows)
            for c in range(self.cols)
            if isinstance(self.matrix[r][c], str)
        )

        return len(visited) == total_pieces

#Artificial Inteligence  A Morden Aproach pag 171
def ac3(board, domains):
    queue = deque()
    for region in domains:
        for neighbor in board.adjacent_regions(region):
            queue.append((region, neighbor))

    while queue:
        xi, xj = queue.popleft()
        if revise( domains, xi, xj):
            if not domains[xi]:
                return False
            for xk in board.adjacent_regions(xi):
                if xk != xj:
                    queue.append((xk, xi))
    return True

def revise( domains, xi, xj):
    revised = False
    new_domain = []

    for val_i in domains[xi]:
        conflict = True
        for val_j in domains[xj]:
            if not is_conflicting( val_i, val_j): 
                conflict = False
                break
        if not conflict:
            new_domain.append(val_i)
        else:
            revised = True

    domains[xi] = new_domain
    return revised

def is_conflicting( val1, val2):
    piece1, variant1, position1 = val1
    piece2, variant2, position2 = val2

    # se as pecas sao diferentes nao ha conflito
    if piece1.label != piece2.label:
        return False

    # coordenadas ocupadas por cada peça
    cells1 = set()
    for i in range(variant1.shape[0]):
        for j in range(variant1.shape[1]):
            if variant1[i][j]:
                row = position1[0] + i
                col = position1[1] + j
                cells1.add((row, col))

    cells2 = set()
    for i in range(variant2.shape[0]):
        for j in range(variant2.shape[1]):
            if variant2[i][j]:
                row = position2[0] + i
                col = position2[1] + j
                cells2.add((row, col))

    # verific aadjacecnia ortogonal entre celulas diferentes
    for (r1, c1) in cells1:
        for (r2, c2) in cells2:
            if abs(r1 - r2) + abs(c1 - c2) == 1:
                return True  # conflip peças adjacentes com o mesmo label

    #verifica blocos 2x2 formados pelas duas pecas 
    combined_cells = cells1.union(cells2)
    for (r, c) in combined_cells:
        square_cells = [
            (r, c),
            (r + 1, c),
            (r, c + 1),
            (r + 1, c + 1)
        ]
        if all(cell in combined_cells for cell in square_cells):
            return True  # confilot formam um bloco 2x2 com o mesmo label

    return False  # nenhum conflito detetad 



#STATE
class NuruominoState:
    state_id = 0
    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id
        self.regions = list(board.regions.keys())
        self.domains = self.compute_domains(board)
        NuruominoState.state_id += 1

    #calcula os dominios de acoes para a board do destado atual
    def compute_domains(self, board):
        domains = {}
        for region_id in board.regions:
            domains[region_id] = []
            for piece in TETROMINOS.values():
                for variant in piece.variants():
                    positions = board.region_fits_tetronimo(region_id, piece, variant)
                    for pos in positions:
                        domains[region_id].append((piece, variant, pos))
        return domains

    def __lt__(self, other):
        return self.id < other.id

class Nuruomino(Problem):
    def __init__(self, initial_board):
        self.initial= NuruominoState(initial_board)
        self.visited_states = set()

    def actions(self, state):
        actions = []

        #estado ja tinha sido visitado 
        if  state.board.key() in self.visited_states:
            return actions 
      

        #se nao adiciona o estado aos visitidados
        self.visited_states.add(state.board.key())

        #aplica o algoritmos ac3 para reduzir os dominios de acoes
        #varifica se o problema a consistencia de arcos dos dominios
        is_arc_consistent = ac3(state.board, state.domains)


        #o problem nao e consistente
        #existemm conflitos de restricoes no dominio
        if not is_arc_consistent:
            return[]

        #e nessario para o programa continuar em caso de boards completametne preenchidas!!
        regions = [r for r in state.domains]
        if not regions:
            return []
     
        # selcionar a região com menor domínio (MRV) e em caso de empate maior grau (mais regioes adjacenttes)
        next_region = None
        best_score = None

        for region in regions:
            domain_size = len(state.domains[region])

            # Contar vizinhos não atribuídos (grau)
            unassigned_neighbors = 0
            for neighbor in state.board.adjacent_regions(region):
                if neighbor in state.domains:
                    unassigned_neighbors += 1

            score = (domain_size, -unassigned_neighbors)

            if best_score is None or score < best_score:
                best_score = score
                next_region = region 
      

        for (piece, variant, pos) in state.domains[next_region]:
            new_board=state.board.place_piece_at_pos(next_region,piece,variant,pos)
            #forward checking por estados invalidos
            if not  new_board.has_isolated_piece()  and not new_board.has_isolated_regions() and new_board.key() not in self.visited_states:
                actions.append((next_region, piece, variant, pos))

        return actions 

    def result(self, state, action):
        region_id, piece, variant, pos = action

        # aplicar a ação
        new_board = state.board.place_piece_at_pos(region_id, piece, variant, pos)
        new_state = NuruominoState(new_board)
        return new_state
 

    def h(self, node):
        pass

    def goal_test(self, state):
        return state.board.is_valid_nurikabe()

def fill_nones(matrix1, matrix2):
    result = []
    for row1, row2 in zip(matrix1, matrix2):
        new_row = []
        for val1, val2 in zip(row1, row2):
            new_row.append(val2 if val1 is None else val1)
        result.append(new_row)
    return result



if __name__ == "__main__":
    TETROMINOS = [
        Tetromino("L", [[1, 0], [1, 0], [1, 1]]),
        Tetromino("I", [[1], [1], [1], [1]]),
        Tetromino("T", [[1, 1, 1], [0, 1, 0]]),
        Tetromino("S", [[0, 1, 1], [1, 1, 0]]),
    ]

    TETROMINOS = {t.label: t for t in TETROMINOS}
    board = Board.parse_instance()
    # board.print_instance()
   
    problem = Nuruomino(board)

    goal_node = breadth_first_tree_search(problem)
    # print(goal_node.state.board.matrix)


    goal = fill_nones(goal_node.state.board.matrix,  board.matrix)
    goal_node.state.board.matrix=goal
    goal_node.state.board.print_instance()

