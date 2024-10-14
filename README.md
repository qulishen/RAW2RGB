## RAW 文件转 RGB 图像
### 由于 RAW 格式有很多种不同的排列，因此需要多次尝试才能确定正确的转换方法。
修改文件路径后运行

python main.py


如果颜色不对，请修改main.py中的裁剪方式 （21行），四种方式皆可尝试；
```python
  # 裁剪
  raw_image = raw_image[:,:]
  # raw_image = raw_image[:,1:-1]
  # raw_image = raw_image[1:-1,:1:-1]
  # raw_image = raw_image[:,:]
```

如果还是不行，请将 datasets/process.py 里的
```python 
# rgb_wb_ccm = apply_ccms(rgb_wb, cam2rgbs)
rgb_wb_ccm = rgb_wb
```

改为
```python 
rgb_wb_ccm = apply_ccms(rgb_wb, cam2rgbs)
# rgb_wb_ccm = rgb_wb
```
并再次尝试四种裁剪方式
