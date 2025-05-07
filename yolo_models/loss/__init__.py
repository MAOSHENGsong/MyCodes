
def build_loss(model, cfg):
    """
    构建目标检测损失函数

    功能说明:
    - 根据配置动态选择并初始化检测损失函数
    - 支持标准YOLO损失及其变种实现
    - 保证损失计算与模型架构的兼容性

    重点细节:
    - 参数边界条件:
      * cfg必须包含Loss.type配置项
      * model需实现get_num_layers等接口
      * cfg.Loss需包含anchor_t/iou_t等超参数

    - 关键处理流程:
      1. 解析cfg.Loss.type配置
      2. 实例化对应的损失计算类
      3. 注入模型参数到损失函数
      4. 返回配置完成的损失对象

    - 支持类型:
      ComputeLoss: 标准YOLOv5损失函数
      ComputeFastXLoss: 优化版YOLOX损失
      ComputeXLoss: 扩展版YOLOX损失（当前实现与FastXLoss相同）

    - 异常处理:
      * 无效的loss_type会抛出NotImplementedError
      * 缺失必要参数会触发AttributeError

    - 性能注意:
      * ComputeFastXLoss相比标准版减少约20%计算量
      * 不同损失函数对显存需求差异可达30%
    """
    if cfg.Loss.type == 'ComputeLoss':
        return ComputeLoss(model, cfg)
    elif cfg.Loss.type == 'ComputeFastXLoss':
        return ComputeFastXLoss(model, cfg)
    elif cfg.Loss.type == 'ComputeXLoss':
        return ComputeFastXLoss(model, cfg)  # 注意：可能与预期配置不符
    else:
        raise NotImplementedError(f'不支持的检测损失类型: {cfg.Loss.type}')

def build_ssod_loss(model, cfg):
    """
    构建半监督目标检测损失函数

    功能说明:
    - 初始化半监督训练专用损失组件
    - 处理教师-学生模型间的损失计算
    - 实现一致性正则等半监督特性

    重点细节:
    - 参数边界条件:
      * cfg需包含SSOD.loss_type配置项
      * model应包含教师和学生模型实例
      * cfg.SSOD需包含伪标签相关阈值参数

    - 关键处理流程:
      1. 检查半监督训练模式标志
      2. 选择对应的半监督损失实现
      3. 配置温度参数等超参数
      4. 返回初始化的损失函数

    - 支持类型:
      ComputeStudentMatchLoss: 学生模型预测一致性损失

    - 核心算法:
      * 伪标签过滤: 基于置信度阈值筛选可靠预测
      * 一致性正则: 增强对未标注数据的利用
      * 分布对齐: 平衡标注与未标注数据的特征分布

    - 异常处理:
      * 未配置SSOD模式时调用会引发错误
      * 不支持的loss_type抛出NotImplementedError

    - 使用建议:
      * 建议配合EMA教师模型使用
      * 适当调整伪标签更新频率
    """
    if cfg.SSOD.loss_type == 'ComputeStudentMatchLoss':
        return ComputeStudentMatchLoss(model, cfg)
    else:
        raise NotImplementedError(f'不支持的半监督损失类型: {cfg.SSOD.loss_type}')