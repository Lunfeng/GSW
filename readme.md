# GSW

## 快速演示
运行示例脚本：
```bash
python extern/Cutie/scripting_demo.py --image_path data/truck/images --mask_path data/truck/mask.png --output_path data/truck/masks
```

## 生成济南2021-2025价格走势示例图
本仓库新增了一个用于生成济南住宅均价示例趋势图的脚本，默认数据为示例值，可根据真实行情调整。

### 运行步骤
1. 建议使用 Python 3.10+，并在仓库根目录创建虚拟环境：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
   ```
2. 安装依赖（当前脚本仅需 matplotlib）：
   ```bash
   pip install matplotlib
   ```
3. 运行脚本生成图表（可指定输出路径）：
   ```bash
   python jinan_price_trend.py --output outputs/jinan_price_trend.png
   ```

修改 `build_price_points` 中的数值即可替换为真实观测数据，生成的图表会包含最低点、最高点标记以及清晰的时间刻度。
