class Callbacks:
    """🔌 YOLOv5回调处理器，集中管理训练生命周期各阶段的钩子函数

    核心作用:
        - 为训练/验证/保存等关键节点提供扩展点
        - 通过注册机制实现模块解耦
        - 支持20+预定义事件节点的回调触发

    回调事件分类:
        ▶ 训练流程
        - on_pretrain_routine_start: 预训练准备开始
        - on_train_start: 训练循环开始
        - on_train_batch_start: 单个batch训练前
        - on_train_epoch_end: epoch训练结束

        ▶ 验证流程
        - on_val_start: 验证循环开始
        - on_val_image_end: 单张图像验证完成
        - on_val_end: 验证完成

        ▶ 系统级事件
        - on_model_save: 模型保存时
        - on_train_end: 训练完全结束
        - teardown: 资源清理

    实现特点:
        - 使用字典维护事件与回调的映射关系
        - 每个事件对应回调函数列表(支持多个处理器)
        - 严格的事件触发顺序控制(代码层面硬编码顺序)

    使用规范:
        1. 通过add方法注册回调函数
        2. 回调函数需匹配事件签名参数
        3. 事件名称需完全匹配预定义键值
    """

    # Define the available callbacks
    _callbacks = {
        'on_pretrain_routine_start': [],
        'on_pretrain_routine_end': [],

        'on_train_start': [],
        'on_train_epoch_start': [],
        'on_train_batch_start': [],
        'optimizer_step': [],
        'on_before_zero_grad': [],
        'on_train_batch_end': [],
        'on_train_epoch_end': [],

        'on_val_start': [],
        'on_val_batch_start': [],
        'on_val_image_end': [],
        'on_val_batch_end': [],
        'on_val_end': [],

        'on_fit_epoch_end': [],  # fit = train + val
        'on_model_save': [],
        'on_train_end': [],

        'teardown': [],
    }

    def register_action(self, hook, name='', callback=None):
        """🔌 注册回调动作到指定钩子

        参数:
            hook: 预定义的钩子名称(必须存在于_callbacks)
            name: 动作标识名(可选，用于调试追踪)
            callback: 要执行的回调函数(必须可调用)

        核心作用:
            - 将自定义回调函数绑定到训练生命周期事件
            - 支持多个回调按注册顺序执行

        重点细节:
            - 严格校验hook有效性，防止错误事件注入
            - 回调存储为字典格式，保留名称引用
            - 使用断言(非异常捕获)确保参数有效性
            - 回调顺序影响执行优先级(先注册先执行)

        参数验证:
            1. hook必须存在于_callbacks预定义键
            2. callback必须是可调用对象
            3. name默认空字符串但建议命名便于调试

        数据结构:
            self._callbacks[hook] = [{'name': 'log', 'callback': func}, ...]
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({'name': name, 'callback': callback})

    def get_registered_actions(self, hook=None):
        """🔍 获取已注册回调的查询接口

        参数:
            hook: 指定钩子名称(可选)，默认返回全部回调

        返回:
            -> 指定hook对应的回调列表
            -> 或完整回调字典(当hook=None时)

        设计用途:
            - 调试时查看回调注册情况
            - 动态获取当前生效的生命周期处理器
            - 支持选择性查询特定阶段回调
        """
        if hook:
            return self._callbacks[hook]
        else:
            return self._callbacks

    def run(self, hook, *args, **kwargs):
        """⚡️ 触发指定钩子的所有回调执行

        参数:
            hook: 要触发的生命周期钩子名称
            *args: 透传给回调的位置参数
            **kwargs: 透传给回调的关键字参数

        执行逻辑:
            1. 校验hook有效性
            2. 按注册顺序同步执行回调
            3. 透传YOLOv5引擎的上下文参数

        关键要求:
            - 回调函数需兼容参数签名
            - 前置hook的异常会影响后续回调
            - 无返回值设计，侧重过程控制

        典型应用:
            - 在训练开始时触发日志初始化
            - 在batch结束时触发进度更新
            - 模型保存时触发云存储上传
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"

        for logger in self._callbacks[hook]:
            logger['callback'](*args, **kwargs)