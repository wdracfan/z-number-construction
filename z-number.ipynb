{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "7vQZTpIPc5ch"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import scipy\n",
    "from tqdm import tqdm\n",
    "\n",
    "import warnings\n",
    "warnings.simplefilter(action='ignore', category=scipy.integrate.IntegrationWarning)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7kDbkezUv6CX"
   },
   "source": [
    "Класс для работы с трапециевидными нечёткими множествами $\\text{FS}(a,b,c,d)$:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "-y2yUMFSeCs5"
   },
   "outputs": [],
   "source": [
    "def trapezoid_mf(a, b, c, d):\n",
    "  '''\n",
    "  Вспомогательная функция для получения функции принадлежности трапециевидного НМ.\n",
    "  '''\n",
    "  return lambda x: \\\n",
    "    0 if x <= a or x >= d else \\\n",
    "    (x - a) / (b - a) if a < x <= b else \\\n",
    "    1 if b <= x <= c else \\\n",
    "    (d - x) / (d - c) if c < x <= d else \\\n",
    "    None\n",
    "\n",
    "class FuzzySet:\n",
    "  def __init__(self, a, b, c, d, u_min=None, u_max=None):\n",
    "    '''\n",
    "    a,b,c,d - значения, задающие НМ FS(a,b,c,d)\n",
    "    [u_min, u_max] - универсум, по умолчанию [a,d]\n",
    "    '''\n",
    "    self.mf = trapezoid_mf(a, b, c, d)\n",
    "    self.min = u_min if u_min != None else a\n",
    "    self.max = u_max if u_max != None else d\n",
    "    self.a, self.b, self.c, self.d = a, b, c, d\n",
    "\n",
    "  def plot(self):\n",
    "    '''\n",
    "    Функция для построения графика функции принадлежности НМ.\n",
    "    '''\n",
    "    xs = np.linspace(self.min, self.max, 100)\n",
    "    ys = [self.mf(x) for x in xs]\n",
    "    plt.plot(xs, ys)\n",
    "    plt.xlim(self.min, self.max)\n",
    "\n",
    "  def specificity(self):\n",
    "    '''\n",
    "    Функция для расчёта специфичности НМ.\n",
    "    '''\n",
    "    return 1 - (self.c + self.d - (self.a + self.b)) / (2 * (self.max - self.min))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "wST8b7MZwCp8"
   },
   "source": [
    "Функция для построения гистограммы (как функции!) по набору данных:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "6vSGpQBUjXir"
   },
   "outputs": [],
   "source": [
    "def histogram(data, bins):\n",
    "  '''\n",
    "  data - набор данных\n",
    "  bins - количество интервалов или строка sturges, sqrt и т.п.\n",
    "\n",
    "  Возвращает функцию-гистограмму.\n",
    "  '''\n",
    "  hist, bin_edges = np.histogram(data, bins, density=True)\n",
    "  def h(x):\n",
    "    for i in range(len(hist)):\n",
    "      if bin_edges[i] <= x < bin_edges[i + 1]:\n",
    "        return hist[i]\n",
    "    return 0\n",
    "  return h"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "q-a4sVLmNXEV"
   },
   "source": [
    "Функция для построения B-части Z-числа при фиксированной A-части (согласно подходу, предложенному в разделе 4.1 работы):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "7nFvppsPNez0"
   },
   "outputs": [],
   "source": [
    "def calculate_b_part(A: FuzzySet, data, bins_min=3, bins_max=None, bins_step=1, bins_best=None, only_peak=False) -> FuzzySet:\n",
    "  '''\n",
    "  A - данная A-часть Z-числа\n",
    "  data - набор данных\n",
    "\n",
    "  Строятся гистограммы с разбиением на k интервалов, где k перебирается от bins_min до bins_max с шагом bins_step.\n",
    "  По умолчанию bins_min = 3, bins_max = len(data), bins_step = 1.\n",
    "\n",
    "  bins_best - наилучшее количество интервалов в гистограмме, допускаются также значения sturges и sqrt.\n",
    "  Должно быть выполнено bins_min <= bins_best <= bins_max или bins_best in ['sturges', 'sqrt'].\n",
    "  По умолчанию bins_best = 'sturges'.\n",
    "\n",
    "  Возвращает B-часть Z-числа, соответствующую данной A-части\n",
    "  '''\n",
    "  if bins_max == None:\n",
    "    bins_max = len(data)\n",
    "  if bins_best == None or bins_best == 'sturges':\n",
    "    bins_best = 1 + np.ceil(np.log(len(data)) / np.log(2))\n",
    "  elif bins_best == 'sqrt':\n",
    "    bins_best = np.ceil(np.sqrt(len(data)))\n",
    "  bins_best = int(bins_best)\n",
    "\n",
    "  assert(bins_min <= bins_best <= bins_max)\n",
    "\n",
    "  a, b = min(data.min(), A.min), max(data.max(), A.max)\n",
    "\n",
    "  if only_peak:\n",
    "    c = scipy.integrate.quad(lambda x: A.mf(x) * histogram(data, bins_best)(x), a=a, b=b)[0]\n",
    "    return FuzzySet(c, c, c, c, 0, 1)\n",
    "\n",
    "  l, c, r = None, None, None\n",
    "  for bins in range(bins_min, bins_max, bins_step):\n",
    "    similarity = scipy.integrate.quad(lambda x: A.mf(x) * histogram(data, bins)(x), a=a, b=b)[0]\n",
    "    if l == None or similarity < l:\n",
    "      l = similarity\n",
    "    if r == None or similarity > r:\n",
    "      r = similarity\n",
    "    if bins == bins_best:\n",
    "      c = similarity\n",
    "\n",
    "  if c == None:\n",
    "    c = scipy.integrate.quad(lambda x: A.mf(x) * histogram(data, bins_best)(x), a=a, b=b)[0]\n",
    "\n",
    "  return FuzzySet(l, c, c, r, 0, 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "bla85PcpLuJN"
   },
   "source": [
    "Класс для построения Z-чисел по набору данных:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "C-M6CmmRMeB0"
   },
   "outputs": [],
   "source": [
    "class ZNumber:\n",
    "  def __init__(self):\n",
    "    pass\n",
    "\n",
    "  def fit(self, data,\n",
    "          bins_min=3, bins_max=None, bins_step=1, bins_best=None,\n",
    "          u_min=None, u_max=None, u_step=None,\n",
    "          optimize='specificity', beta=0.5, s_threshold=0.5, c_threshold=0.7,\n",
    "          defuzzify='peak'):\n",
    "    '''\n",
    "    Описания параметров bins_* см. в комментарии к функции calculate_b_part.\n",
    "\n",
    "    Перебор значений a,b,c,d, задающих A-часть FS(a,b,c,d), производится на отрезке [u_min, u_max] с шагом u_step.\n",
    "    По умолчанию u_min = min(data), u_max = max(data), u_step = (u_max - u_min) / 10.\n",
    "\n",
    "    optimize: 'specificity', 'b' или 'both' - что устремляется к максимуму (см. раздел 4.2 работы)\n",
    "    beta, s_threshold, c_threshold - параметры оптимизации (см. раздел 4.2 работы)\n",
    "\n",
    "    defuzzify: 'centroid' или 'peak' - дефаззификация B-части методом центроидов или по координате вершины\n",
    "    '''\n",
    "    if u_min == None:\n",
    "      u_min = min(data)\n",
    "    if u_max == None:\n",
    "      u_max = max(data)\n",
    "    if u_step == None:\n",
    "      u_step = (u_max - u_min) / 10\n",
    "\n",
    "    best_score = None\n",
    "    best_subscore = None\n",
    "    best_A = None\n",
    "\n",
    "    for a in tqdm(np.linspace(u_min, u_max, int((u_max - u_min) / u_step) + 1)):\n",
    "      for b in np.linspace(a, u_max, int((u_max - a) / u_step) + 1):\n",
    "        for c in np.linspace(b, u_max, int((u_max - b) / u_step) + 1):\n",
    "          for d in np.linspace(c, u_max, int((u_max - c) / u_step) + 1):\n",
    "            A = FuzzySet(a, b, c, d, u_min, u_max)\n",
    "            specificity = A.specificity()\n",
    "\n",
    "            if optimize == 'b' and specificity < s_threshold:\n",
    "              continue\n",
    "\n",
    "            B = calculate_b_part(A, data, bins_min, bins_max, bins_step, bins_best, only_peak=(defuzzify == 'peak'))\n",
    "            b_defuzzified = (B.a + B.b + B.d) / 3 if defuzzify == 'centroid' else B.b\n",
    "\n",
    "            if optimize == 'specificity' and b_defuzzified < c_threshold:\n",
    "              continue\n",
    "\n",
    "            if optimize == 'specificity':\n",
    "              score = specificity\n",
    "              subscore = b_defuzzified\n",
    "            elif optimize == 'b':\n",
    "              score = b_defuzzified\n",
    "              subscore = specificity\n",
    "            elif optimize == 'both':\n",
    "              score = beta * b_defuzzified + (1 - beta) * specificity\n",
    "              subscore = score\n",
    "\n",
    "            if best_score == None or best_score < score or best_score == score and best_subscore < subscore:\n",
    "              best_score = score\n",
    "              best_subscore = subscore\n",
    "              best_A = A\n",
    "\n",
    "    self.A = best_A\n",
    "    self.B = calculate_b_part(best_A, data, bins_min, bins_max, bins_step, bins_best, only_peak=False)\n",
    "\n",
    "  def plot(self):\n",
    "    print(f\"A = FS({self.A.a}, {self.A.b}, {self.A.c}, {self.A.d})\")\n",
    "    print(f\"B = FS({self.B.a}, {self.B.b}, {self.B.d})\")\n",
    "    plt.figure(figsize=(10,5))\n",
    "\n",
    "    plt.subplot(1, 2, 1)\n",
    "    self.A.plot()\n",
    "    plt.title('A-часть')\n",
    "\n",
    "    h = np.histogram(data, density=True)\n",
    "    plt.stairs(h[0] / max(h[0]), h[1], fill=True, color=(1, 0.5, 0, 0.4))\n",
    "\n",
    "    plt.subplot(1, 2, 2)\n",
    "    self.B.plot()\n",
    "    plt.ylim(0, 1.05)\n",
    "    plt.title('B-часть')\n",
    "\n",
    "    plt.show()"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
