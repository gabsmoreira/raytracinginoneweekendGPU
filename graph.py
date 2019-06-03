import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

# normal
# 200x100 = 0.0129257 s
# 500x250 = 0.0794398 s  
# 1000x500 = 0.325962 s
# 1600x800 = 0.854987 s
# 2000x1000 = 1.30089 s
# 2400x1200 = 1.85436 s
# 3000x1500 = 2.85305 s
# 4000x2000 = 5.07813 s
# 5000x2500 = 7.92686 s
# 8000x4000 = 20.3372 s

# gpu

# 200x100 = 2.9374 s
# 500x250 = 2.47874 s
# 1000x500 = 2.90023 s
# 1600x800 = 3.26873 s
# 2000x1000 = 3.3477 s
# 2400x1200 = 3.784 s
# 3000x1500 = 4.28702 s
# 4000x2000 = 5.38314 s
# 5000x2500 = 6.40131 s
# 8000x4000 = 12.951 s

plt.xlabel('Tamanho da imagem em pixels')
plt.ylabel('Tempo para processamento do ray tracing em segundos')
pixels = [20000, 125000, 500000, 1280000, 2000000, 2880000, 4500000, 8000000, 12500000, 32000000]

normal = [0.0129257, 0.0794398, 0.325962, 0.854987, 1.30089, 1.85436, 2.85305, 5.07813, 7.92686, 20.3372]

gpu = [2.9374, 2.47874, 2.90023, 3.26873, 3.3477, 3.784, 4.28702, 5.38314, 6.40131, 12.951]

plt.plot(pixels, normal, label='normal')
plt.plot(pixels, gpu, label='CUDA')
plt.legend(loc='upper left')
plt.title('Diferença do processamento do método de Ray tracing usando a CUDA')
plt.show()