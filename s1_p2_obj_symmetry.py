# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

# KASAL is an interactive rotational symmetry analysis software that supports eight types of rotational symmetries.
# KASAL 是一个交互式旋转对称分析软件，支持 8 种旋转对称类型。

from kasal.app.polyscope_app import app # pip install kasal-6d

if __name__ == '__main__':

    # Set the folder path; KASAL will automatically search for all PLY or OBJ files in the folder
    # 设置文件夹路径，KASAL 会自动查找文件夹下所有 PLY 或 OBJ 文件
    mesh_path = 'gearbox-picking'

    # Launch the graphical user interface (GUI) of KASAL
    # 启动 KASAL 的图形界面
    app(mesh_path)

    pass

'''
1. C(=1) : Circular Item（圆形物体）
对称类型：C₁（无旋转对称性，或只有恒等操作）
物理含义：物体在绕轴旋转 360° 时，没有任何非平凡的旋转对称（除了不转）。
但为什么叫 “Circular Item”？这里可能是指物体外形接近圆形，但实际对称性很低，或者特指只有反射对称而无旋转对称的薄圆盘（类似于 C₁ 但带有镜面）。
不过更常见的解释是：C₁ 表示无对称性，但此处被标注为 Circular Item，可能来自特定软件对一类“圆柱/圆形零件”的归类。

2. D(>1) : n-fold Prismatic Item（n 棱柱形物体）
对称类型：Dₙ（二面体群，n ≥ 2）
物理含义：物体具有一个 n 重旋转轴，并且还有垂直于该轴的 n 个二重旋转轴（或反射面）。典型例子：正 n 棱柱（如六角铅笔、方柱）。D(>1) 表示 n 大于 1，即至少有 2 重旋转对称。


3. D(=1) : n-fold Pyramidal Item（n 棱锥形物体）
对称类型：Cₙᵥ（锥形对称群，包含一个 n 重轴和 n 个垂直的镜面）
物理含义：写为 D(=1) 可能是符号误用或特例，实际上应理解为n 棱锥（如四角锥）。它有一个 n 重旋转轴，但没有垂直于主轴的二重轴，而是有多个通过主轴的镜面。常见于金字塔形、圆锥形物体。


4. C(>1) : Cylindrical Item（圆柱形物体）
图片中的 Note 说：if n < 2, n = C(>1): Cylindrical Item，意思是当 n（旋转阶数）小于 2 时，定义为 C(>1) 圆柱形物体。这有点反直觉。实际上，真正的圆柱具有无限重旋转对称（C∞），但这里 C(>1) 可能泛指连续旋转对称（任意角度旋转不变）的物体，如圆柱、圆锥、球体的一部分。

5. C(>>1) : Spherical Item（球形物体）
图片中 Setting of D(>1) a C(>>1): Spherical Item，意思是当 D(>1) 或 C(>>1) 时，视为球形物体。C(>>1) 可理解为无限重旋转对称（如球体，在任何轴向上旋转任意角度都不变）。球形物体是所有方向旋转对称的极端情况。

6. P(4) : Tetrahedral Item（四面体形物体）
对称类型：Tₓ（四面体群，阶数 12）

物理含义：物体具有四面体的全部对称操作（4 个 3 重轴、3 个 2 重轴等）。例子：正四面体。

7. P(8) : Octahedral Item（八面体形物体）
对称类型：Oₕ（八面体/立方体群，阶数 24）

物理含义：具有立方体或八面体的全部对称性（3 个 4 重轴、4 个 3 重轴、6 个 2 重轴等）。例子：正八面体、立方体。

8. P(20) : Icosahedral Item（二十面体形物体）
对称类型：Iₕ（二十面体/十二面体群，阶数 60）

物理含义：具有正二十面体或正十二面体的对称性（6 个 5 重轴、10 个 3 重轴、15 个 2 重轴）。例子：足球（C₆₀）分子结构、正二十面体。


'''
