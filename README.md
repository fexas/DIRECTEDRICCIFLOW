# DirectedRicciFlow

（更新日期：2024.03.22）

本仓库为本人2024年在中国人民大学进行的本科毕设“有向图上的Ricci曲率及应用”中应用部分的对应的代码

## 摘要
前提：毕设的理论部分做了如下四部分工作：

​	1.定义了有向图上的Lin-Lu-Yau Ricci曲率，证明了其存在性。

​	2.基于上述曲率定义了有向图上的Ricci flow，且证明其对应微分方程租解的存在性。

​	3.证明了有向图上Lin-Lu-Yau Ricci曲率存在基于*-耦合（*​-coupling）的形式

​	4.定义了离散化的Ricci flow，为后续的应用奠定基础

本仓库实现的为有向图上的Ricci flow在 社区检测（community detection）中的初步应用。

运行方式：

```bash
cd DIRECTEDRICCIFLOW/DirectedRicciFLow/
./doit.sh
```

![基于有向图上Ricci流的社区检测](https://github.com/fexas/DIRECTEDRICCIFLOW/blob/main/gif/ricci-flow-animation.gif)
