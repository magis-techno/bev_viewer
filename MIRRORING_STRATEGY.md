# 镜像翻转方案说明（配合 `bev_debug_viewer` 项目）

本文档说明：如果要把左舵国家数据“等效”为右舵国家风格的数据，推荐如何在**不大改现有数据资产**的前提下，把镜像处理接入到数据加载与可视化链路中。

这份说明**不要求先修改训练网络**，而是先把“数据表示、翻转接口、可视化验收”三件事做对。当前项目 `bev_debug_viewer.py` 和 `generate_synthetic_scene.py` 就是这个思路的最小原型。

---

## 1. 总体思路

推荐采用两阶段方案：

### 阶段 A：先在 dataloader / sample 构造阶段做在线镜像

先不要离线重写整套数据资产，不要一开始就去批量生成一份“右舵版”图像、LiDAR、地图、标定、pose 文件。

优先做法是：

- 原始数据保持不变
- 每次读取一个 sample 后，在内存里构造一个 **mirrored sample**
- 训练或调试时，网络/可视化只消费这份镜像后的 sample

这样做的好处：

- 软件改动最小
- 不污染原始数据资产
- 最容易做 A/B 对照和可视化验收
- 发现 bug 时回滚最简单
- 哪些量必须翻、哪些量不需要翻，边做边能看清楚

### 阶段 B：确认在线镜像有效后，再把热点步骤离线化

在线镜像跑通后，再对整条链路做 profile，找出真正耗时的部分。然后只把这些**稳定、纯函数、重复计算多**的步骤离线预存。

典型策略：

- 先保证**数学正确**
- 再做**性能优化**
- 不要在镜像逻辑还没验明白之前，就提前把大量中间量固化到磁盘

这是本文档最核心的建议。

---

## 2. 为什么不建议一开始就“全量离线翻转”

从工程角度看，“图像、LiDAR、地图、pose、标定、bbox 全都预先翻一遍”有 4 个明显问题：

1. **资产会膨胀**  
   数据量翻倍，管理成本也会上升。

2. **一致性容易出错**  
   某些文件翻了，某些没翻；或者图像翻了但标定没同步；或者 bbox/mask/map 在不同坐标系里翻得不一致。

3. **不利于调试**  
   看到问题时，难以判断 bug 是在原始数据、离线转换脚本，还是训练读取逻辑。

4. **很多东西其实不需要翻成新文件**  
   例如一些只在 sample 内部临时使用的张量，完全可以在线处理。

因此，建议先把“镜像”视为一个**sample 级别的变换**，而不是“资产重建工程”。

---

## 3. 这个项目里建议固定下来的接口抽象

当前项目里已经有一个通用的场景接口 `SceneRecord`，由 `parse_scene()` 解析得到，里面包含：

- `ego_T_world`
- `cameras`
- `lidar`
- `boxes`
- `map_record`

对应代码位置：

- `bev_debug_viewer.py` 中的 `SceneRecord`
- `CameraRecord`
- `LidarRecord`
- `BoxRecord`
- `MapRecord`

这套抽象已经足够接近我们真正需要的“统一翻转入口”。

推荐把镜像接口固定成：

```python
scene_m = mirror_scene(scene)
```

或者面向对象一点：

```python
scene_m = scene.mirrored()
```

无论以后接 LSS 风格 dataloader，还是接你们自己的 camera+LiDAR 融合链路，都建议保持这个接口不变。

---

## 4. 什么东西应该被镜像

镜像的本质，是在**同一帧场景语义不变**的前提下，把所有与 ego 左右方向相关的量做一个一致的反射。

常见规则是：

- 坐标：`(x, y, z) -> (x, -y, z)`
- 平面朝向：`yaw -> -yaw`
- 图像：水平翻转
- 左/右语义标签：必要时交换

具体到这个项目中的各个对象：

### 4.1 相机图像

如果网络输入是原始相机图像，那么**图像水平翻转是前提**。

原因很简单：

- 相机外参、ego 几何、bbox、地图都镜像了
- 但图像内容如果没翻，视觉证据还是原来的左舵世界
- 这样视觉和几何不自洽

所以 camera-to-BEV 场景下，不能只改相机矩阵，不改图像。

### 4.2 相机外参

相机图像翻转后，相机几何也必须同步镜像。

在这个项目的抽象里，就是修改 `CameraRecord.T_cam_to_ego`。

建议统一由一个函数处理，例如：

```python
T_cam_to_ego_m = mirror_camera_extrinsic(T_cam_to_ego)
```

### 4.3 LiDAR 点云

LiDAR 通常不需要重新生成点云文件；在 sample 被读入内存之后，直接对 ego frame 下的点做：

```python
[x, y, z] -> [x, -y, z]
```

即可。

如果 LiDAR 分支已经先变成了体素或 BEV feature，也可以直接翻转 lateral 维。

### 4.4 3D bbox / 动态目标 GT

bbox 需要同时处理：

- `center_ego[:, 1] *= -1`
- `yaw_rad *= -1`

如果有速度、加速度、轨迹，也同样对 lateral 分量取负。

### 4.5 地图 GT

如果地图以 polyline / lane centerline / 边界线的方式存在，那么也做：

```python
(x, y, z) -> (x, -y, z)
```

如果地图已经 rasterize 成 ego-BEV 图层，也可以直接翻转横向维。

### 4.6 ego pose / 全局 pose

这里要分清楚：

- **训练/调试 sample 内部用到的局部 ego-frame 量**：需要镜像
- **原始日志里的全局 pose 文件**：一开始不建议离线重写

换句话说，先在 sample 级别里做镜像，不要先碰原始资产。

---

## 5. 这个项目里哪些部分适合先在线做

按当前 `bev_debug_viewer` 这个项目，以下步骤适合**直接在线做**：

### 5.1 图像翻转

对 `CameraRecord.image_path` 对应图像读取后做水平翻转。

因为图像本身就是按 sample 读取的，在线翻很直接，而且最方便调试。

### 5.2 LiDAR 点镜像

`load_lidar_points()` 读进来之后，在内存里改 `y` 即可。

### 5.3 bbox 镜像

`BoxRecord` 是很轻的小对象，在线改开销很低。

### 5.4 map polyline 镜像

polyline 通常点数不大，在线变换基本没有性能压力。

### 5.5 可视化重投影

在 `draw_camera_overlay()`、`draw_bev()` 这些函数里直接消费镜像后的 `scene`，就能立即检查结果是否正确。

这正是这个项目最适合拿来做 sanity check 的地方。

---

## 6. 哪些步骤值得后续离线预存

在线镜像跑通后，再分析耗时。一般只建议离线化下面这些“重复计算多、纯函数、结果稳定”的步骤。

### 6.1 图像解码 / 缩放 / 归一化

如果 profile 发现相机图像 IO 和 resize 很慢，可以考虑把：

- 统一分辨率图像
- 归一化后的 tensor
- 或镜像版本图像

做离线缓存。

这是最常见也最可能有效的优化点。

### 6.2 LiDAR 到 ego 的预变换

如果每次都要做：

- 多 sweep 聚合
- sensor frame -> ego frame 变换
- 过滤 / 下采样

而且这部分占比较大，那么可以把“标准化到 ego frame 的点云”离线预存。

之后在线镜像只做一步 `y -> -y`。

### 6.3 地图局部裁剪 / 局部 raster

如果你们后续用了比较重的地图处理，例如：

- 从全局 HD map 查询局部 patch
- 再 rasterize
- 再对齐到 ego

那么这部分也非常适合缓存成 sample 级局部地图。

### 6.4 相机几何中间量

对于 LSS 一类模型，某些与相机内参、图像尺寸、离散深度 bins 强相关、且每帧重复使用的几何中间量，也可以离线缓存。

但建议在**镜像逻辑完全正确之后**再做，不要一开始就缓存，否则调试很痛苦。

---

## 7. 建议的工程推进顺序

推荐按下面 5 步走：

### 第一步：统一 sample 抽象

先固定输入接口，不管真实数据来自哪里，最终都转成统一对象：

- cameras
- lidar
- boxes
- map
- ego pose

当前项目已经基本有了这套结构。

### 第二步：实现单一镜像入口

让镜像逻辑全部收口到一个入口，例如：

```python
scene_m = mirror_scene(scene)
```

不要在不同模块里各翻各的。

### 第三步：先做可视化验收

利用本项目的可视化函数，把下面几类内容画在一起：

- 图像
- LiDAR 投影
- bbox
- 地图
- BEV footprint / frustum

先用眼睛确认几何是对的，再谈训练。

### 第四步：跑 profile

确认每个 sample 的耗时主要在哪：

- 图像读取
- 图像 resize/flip
- LiDAR 预处理
- 地图裁剪/raster
- 几何投影

### 第五步：只离线缓存热点

对 profile 结果里真正占大头的部分做缓存，保留镜像主逻辑仍然在 sample 接口这一层。

---

## 8. 配合当前代码，建议怎么落地

### 8.1 当前项目的角色

当前项目最适合承担两件事：

1. **统一场景接口定义**
2. **镜像正确性的可视化验收**

也就是说，它现在不是训练代码，而是训练前的数据几何调试工具。

### 8.2 推荐增加的函数

后续可在 `bev_debug_viewer.py` 里补这些函数：

```python
mirror_points_y(points_xyz)
mirror_yaw(yaw_rad)
mirror_box(box)
mirror_polyline(poly)
mirror_camera_extrinsic(T_cam_to_ego)
mirror_lidar_extrinsic(T_lidar_to_ego)
mirror_scene(scene)
```

并保证：

- 原始 scene 可视化正常
- `mirror_scene(scene)` 后可视化也正常
- `mirror_scene(mirror_scene(scene))` 近似回到原始结果

这第三条非常重要，是一个很好用的回归测试。

### 8.3 和 LSS 风格的关系

虽然这个项目不直接依赖 LSS，但接口思路可以向 LSS 靠拢：

- 相机图像
- 相机内参
- 相机到 ego 外参
- ego frame 的 GT

这样未来如果要接入 LSS 或类似的 camera-to-BEV dataloader，就不会重构太多。

---

## 9. 一个务实的结论

如果你们现在的目标是：

- 用左舵国家数据模拟右舵国家
- 先验证镜像链路是否靠谱
- 暂时不动训练主干

那么最合理的路径就是：

1. **先在 dataloader / sample 层在线镜像**
2. **通过本项目把图像、LiDAR、bbox、地图、frustum 全画出来做验收**
3. **确认无误后再 profile**
4. **只把重计算环节离线预存**

这是风险最小、迭代最快、最容易定位 bug 的方案。

---

## 10. 建议的下一步

结合当前项目，下一步最值得做的是：

1. 在 `bev_debug_viewer.py` 里正式加入 `mirror_scene(scene)`
2. 增加一个 `--mirror` 命令行参数
3. 输出 `original / mirrored / mirrored_twice` 三组图
4. 对三组结果做自动 sanity check

这样这套工具就不仅能“看图”，还能变成一套稳定的镜像数据验收基座。

