import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd

# carrega os dados de tensão do arquivo CSV
csv_file = "measures.csv"
df = pd.read_csv(csv_file, delimiter=';', decimal=',', header=None)

# extrai os valores de tensão do DataFrame e converte para float
tensões = df.values.astype(float)
vmin = np.min(tensões)
vmax = np.max(tensões)

# gera as coordenadas x e y para a grade
x = np.arange(0, tensões.shape[1] * 2, 2)  # 2 cm entre pontos
y = np.arange(0, tensões.shape[0] * 2, 2)  # 2 cm entre pontos
x_grid, y_grid = np.meshgrid(x, y)

# interpola os valores de tensão para criar uma grade suave
grid_tensões = griddata((x_grid.ravel(), y_grid.ravel()), tensões.ravel(), (x_grid, y_grid), method='cubic')

# calcula o gradiente das linhas equipotenciais para obter Ex e Ey
Ey, Ex = np.gradient(-grid_tensões)  # sinal negativo porque estamos encontrando o gradiente da tensão

# Plota a imagem da distribuição de tensão usando um mapa de cores
plt.figure()
plt.imshow(grid_tensões, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
cbar = plt.colorbar(label='Tensão')
cbar.set_ticks([vmin, vmax])  # ajusta a escala
cbar.set_ticklabels([str(vmin), str(vmax)])
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.title('Distribuição de Tensão')
plt.grid(True)

# Plota apenas linhas equipotenciais
plt.figure()
contour_lines = np.linspace(vmin, vmax, 21)
plt.contour(x_grid, y_grid, grid_tensões, levels=contour_lines, cmap='viridis')
plt.colorbar(label='Tensão (V)')
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.title('Linhas Equipotenciais')
plt.grid(True)

# Plota linhas equipotenciais e vetores de campo elétrico juntos
plt.figure()
plt.contour(x_grid, y_grid, grid_tensões, levels=contour_lines, cmap='viridis')
plt.colorbar(label='Tensão (V)')
plt.quiver(x_grid, y_grid, Ex, Ey, scale=20, color='red')  # observe o sinal negativo para Ey
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.title('Linhas Equipotenciais e Vetores de Campo Elétrico')
plt.grid(True)

plt.show()
