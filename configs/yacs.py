import copy
import io
import logging
import os
from ast import literal_eval
import yaml


# 定义python、YAML配置文件的合法扩展名
_YAML_EXTS = {"", ".yaml", ".yml"}
_PY_EXTS = {".py"}

# 用于检测输入是否为文件对象
_FILE_TYPES = (io.IOBase,)

# 指定CfgNodes允许的数据类型
_VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}

# logging.getLogger()方法的作用是获取或创建一个日志记录器。
# 使用__name__作为参数的好处，即按模块层次结构组织日志，方便过滤和处理。
logger = logging.getLogger(__name__)


class CfgNode(dict):
    """
    #完整继承字典的键值存储和操作方法,通过属性访问 (config.key)、类型校验和嵌套结构支持增强配置管理
    """
    IMMUTABLE = "__immutable__"          # 标记配置是否只读
    DEPRECATED_KEYS = "__deprecated_keys__"  # 存储已废弃键名
    RENAMED_KEYS = "__renamed_keys__"     # 存储键重命名映射
    NEW_ALLOWED = "__new_allowed__"       # 控制是否允许新增配置项

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        # 调用 _create_config_tree_from_dict 方法，将 init_dict 中的嵌套字典递归地转换为 CfgNodes
        # 对象
        init_dict = self._create_config_tree_from_dict(init_dict, key_list)
        # 调用父类的构造函数，将转换后的 init_dict 传递给父类进行初始化
        super(CfgNode, self).__init__(init_dict)
        # 在实例的字典中添加一个名为 IMMUTABLE 的属性，并将其值设置为 False
        self.__dict__[CfgNode.IMMUTABLE] = False
        self.__dict__[CfgNode.DEPRECATED_KEYS] = set()
        self.__dict__[CfgNode.RENAMED_KEYS] = {}
        self.__dict__[CfgNode.NEW_ALLOWED] = new_allowed

    @classmethod
    def _create_config_tree_from_dict(cls, dic, key_list):
        """
        使用给定的字典创建一个配置树。字典内的任何类似字典的对象都将被视为一个新的 CfgNode。
        Args:
            dic (dict):
            key_list (list[str]): 从根节点索引此 CfgNode 的名称列表
        """
        dic = copy.deepcopy(dic)
        for k, v in dic.items():
            if isinstance(v, dict):
                # Convert dict to CfgNode
                dic[k] = cls(v, key_list=key_list + [k])
            else:
                # 从根节点索引此 CfgNode 的名称列表
                _assert_with_logging(
                    _valid_type(v, allow_cfg_node=False),
                    "Key {} with value {} is not a valid type; valid types: {}".format(
                        ".".join(key_list + [k]), type(v), _VALID_TYPES
                    ),
                )
        return dic

    def __getattr__(self, name):
        """
        通过属性访问方式动态获取字典键值，实现点语法访问配置项

        作用：
        - 将字典键转换为对象属性，允许使用`cfg.model.layers`代替`cfg["model"]["layers"]`
        - 严格校验键存在性，访问未定义键时抛出AttributeError（而非KeyError）保持对象访问语义

        重点：
        1. 仅处理不存在于类属性和方法名的属性请求，已有属性（如`keys`/`items`）按正常对象属性访问
        2. 支持嵌套CfgNode访问，通过递归转换实现层级式配置访问（如`cfg.dataset.transform.resize`）
        3. 与__setattr__配合实现写保护，在配置冻结(is_frozen)时禁止修改操作
        """
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        """
        # 主要作用：控制类属性的赋值行为，实现安全属性设置
        # 重点：
        #   - 检查实例是否冻结（immutable），禁止已冻结实例的修改
        #   - 防止覆盖内部__dict__中的属性，保护核心状态
        #   - 验证值类型有效性（通过 _valid_type 校验）
        #   - 最终通过字典方式设置属性（self[name] = value）
        # 注意：
        #   - 必须通过 __dict__ 检查避免递归调用
        #   - 依赖 is_frozen() 状态判断方法
        #   - 使用 _assert_with_logging 替代常规断言
        """
        if self.is_frozen():
            raise AttributeError(
                "Attempted to set {} to {}, but CfgNode is immutable".format(
                    name, value
                )
            )

        _assert_with_logging(
            name not in self.__dict__,
            "Invalid attempt to modify internal CfgNode state: {}".format(
                name),
        )
        _assert_with_logging(
            _valid_type(value, allow_cfg_node=True),
            "Invalid type {} for key {}; valid types = {}".format(
                type(value), name, _VALID_TYPES
            ),
        )

        self[name] = value

    def __str__(self):
        """
        # 主要作用：生成格式化的配置信息字符串，突出层次结构
        # 重点：
        #   - 递归处理嵌套的 CfgNode 对象
        #   - 自动对配置键进行排序输出
        #   - 为嵌套配置添加层级缩进（2空格）
        #   - 区分普通值（单行显示）和嵌套配置（多行显示）
        # 注意：
        #   - 使用内部 _indent 函数处理多行缩进
        #   - 通过 isinstance(v, CfgNode) 检测嵌套配置
        #   - 最终返回没有首行缩进的标准字符串结构
        # 内部辅助函数：处理多行字符串的缩进
        """
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)  # 首行不缩进
            s = [(num_spaces * " ") + line for line in s]  # 后续行添加缩进
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):  # 按键名字母顺序排序
            seperator = "\n" if isinstance(v, CfgNode) else " "  # 嵌套对象换行显示
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)  # 统一添加缩进
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        # 主要作用：生成开发者友好的对象表示形式
        # 重点：
        #   - 组合类名与父类的原生__repr__结果
        #   - 用于调试时明确显示对象类型及内容
        # 注意：与__str__不同，该表示更侧重机器可读性
        return "{}({})".format(self.__class__.__name__,
                               super(CfgNode, self).__repr__())

    def dump(self, **kwargs):
        """
        # 主要作用：将配置对象序列化为YAML格式字符串
        # 重点：
        #   - 递归转换嵌套CfgNode为标准字典结构
        #   - 执行深度类型有效性验证
        #   - 支持yaml.safe_dump的所有参数透传
        # 注意：
        #   - 依赖yaml模块的安全序列化方法
        #   - 转换过程可能抛出类型校验异常
        """

        def convert_to_dict(cfg_node, key_list):
            """
            # 递归转换辅助函数：将CfgNode转换为可序列化的字典
            # 重点：
            #   - 深度遍历嵌套结构
            #   - 对每个节点执行类型校验
            #   - 自动记录当前键路径用于错误定位
            """
            if not isinstance(cfg_node, CfgNode):
                _assert_with_logging(
                    _valid_type(cfg_node),
                    "Key {} with value {} is not a valid type; valid types: {}".format(
                        ".".join(key_list), type(cfg_node), _VALID_TYPES
                    ),
                )
                return cfg_node
            else:
                cfg_dict = dict(cfg_node)
                for k, v in cfg_dict.items():
                    cfg_dict[k] = convert_to_dict(
                        v, key_list + [k])  # 递归处理嵌套配置
                return cfg_dict

        self_as_dict = convert_to_dict(self, [])  # 从根节点开始转换
        return yaml.safe_dump(self_as_dict, **kwargs)  # 透传缩进/编码等格式参数

    def merge_from_file(self, cfg_filename):
        """Load a yaml config file and merge it this CfgNode."""
        with open(cfg_filename, "r", encoding="utf-8") as f:
            cfg = self.load_cfg(f)
        self.merge_from_other_cfg(cfg)

    def merge_from_other_cfg(self, cfg_other):
        """Merge `cfg_other` into this CfgNode."""
        _merge_a_into_b(cfg_other, self, self, [])

    def merge_from_list(self, cfg_list):
        """
        # 主要作用：通过键值对列表（如命令行参数）动态更新配置项
        # 重点：
        #   - 支持点分隔的嵌套键路径（如 "MODEL.HIDDEN_SIZE"）
        #   - 自动处理弃用/重命名的配置键
        #   - 执行值类型校验与强制转换
        #   - 严格验证配置键的合法性
        # 注意：
        #   - 输入列表必须为偶数长度（键值成对）
        #   - 遇到非法键会立即抛出异常中断
        #   - 依赖 _decode_cfg_value 实现值解析
        """

        _assert_with_logging(
            len(cfg_list) % 2 == 0,
            "Override list has odd length: {}; it must be a list of pairs".format(
                cfg_list
            ),
        )
        root = self
        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):  # 遍历键值对
            # 处理已弃用/重命名的配置键
            if root.key_is_deprecated(full_key):
                continue
            if root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)

            # 拆分嵌套键路径（如 "A.B.C" -> ["A", "B", "C"]）
            key_list = full_key.split(".")
            d = self
            # 逐层验证配置结构
            for subkey in key_list[:-1]:
                _assert_with_logging(
                    subkey in d, "Non-existent key: {}".format(full_key)
                )
                d = d[subkey]

            # 最终层键校验与赋值
            subkey = key_list[-1]
            _assert_with_logging(subkey in d,
                                 "Non-existent key: {}".format(full_key))
            value = self._decode_cfg_value(v)  # 解析原始值（如字符串转特定类型）
            value = _check_and_coerce_cfg_value_type(
                value, d[subkey], subkey, full_key)
            d[subkey] = value

    def freeze(self):
        """
        # 主要作用：冻结当前配置节点及其所有子节点，使其不可变
        # 重点：
        #   - 递归锁定所有嵌套的CfgNode对象
        #   - 通过设置_immutable标志位实现全局锁定
        #   - 冻结后任何修改操作将触发AttributeError
        # 注意：
        #   - 通常应在完成配置后调用
        #   - 依赖_immutable内部方法实现状态切换
        """
        self._immutable(True)

    def defrost(self):
        """Make this CfgNode and all of its children mutable."""
        self._immutable(False)

    def is_frozen(self):
        """
        # 主要作用：判断当前配置节点是否处于冻结（不可变）状态
        # 重点：
        #   - 直接访问内部__dict__绕过属性拦截机制
        #   - 通过 IMMUTABLE 常量键值获取状态标志
        #   - 返回布尔值表示当前可修改性
        # 注意：
        #   - 与 freeze() 方法构成状态管理对
        #   - 状态标志存储于对象__dict__的特殊字段
        """
        return self.__dict__[CfgNode.IMMUTABLE]

    def _immutable(self, is_immutable):
        """
        # 主要作用：设置当前节点及所有嵌套子节点的不可变状态
        # 重点：
        #   - 通过__dict__直接操作状态标志，规避属性访问逻辑
        #   - 双重遍历机制：同时处理__dict__属性和字典值中的CfgNode
        #   - 递归传播状态至所有子配置节点
        #   - 使用类常量IMMUTABLE作为状态键保证一致性
        # 注意：
        #   - 该方法是freeze()的实际实现核心
        #   - 递归深度与配置结构复杂度成正比
        #   - 修改后会影响__setattr__的赋值检查"""

        # 直接操作字典避免触发__setattr__
        self.__dict__[CfgNode.IMMUTABLE] = is_immutable
        # 遍历实例属性中的CfgNode
        for v in self.__dict__.values():
            if isinstance(v, CfgNode):
                v._immutable(is_immutable)
        # 遍历字典值中的CfgNode
        for v in self.values():
            if isinstance(v, CfgNode):
                v._immutable(is_immutable)

    def clone(self):
        """Recursively copy this CfgNode."""
        return copy.deepcopy(self)

    def key_is_deprecated(self, full_key):
        # 主要作用：检测指定配置键是否已被标记为弃用
        # 重点：
        #   - 直接访问内部DEPRECATED_KEYS集合进行存在性检查
        #   - 检测到弃用键时自动记录警告日志
        #   - 返回布尔值指示是否需要忽略该键
        # 注意：
        #   - 需使用完整的点分键路径（如 "MODEL.OLD_LAYER"）
        #   - 与merge_from_list的键处理流程直接关联
        """Test if a key is deprecated."""
        if full_key in self.__dict__[CfgNode.DEPRECATED_KEYS]:
            logger.warning(
                "Deprecated config key (ignoring): {}".format(full_key))
            return True
        return False

    def key_is_renamed(self, full_key):
        # 主要作用：检测指定配置键是否已被重命名
        # 重点：
        #   - 查询RENAMED_KEYS字典的键空间
        #   - 纯检测不触发日志，由调用方处理异常
        #   - 返回布尔值指示是否需要后续处理
        # 注意：
        #   - 需配合register_renamed_key使用
        #   - 检测到重命名键时应调用raise_key_rename_error
        """Test if a key is renamed."""
        return full_key in self.__dict__[CfgNode.RENAMED_KEYS]

    def raise_key_rename_error(self, full_key):
        # 主要作用：抛出明确的键重命名异常，引导用户更新配置
        # 重点：
        #   - 从RENAMED_KEYS获取新键名和附加信息
        #   - 支持带说明消息的元组存储格式
        #   - 生成包含新旧键名和迁移指引的完整错误信息
        # 注意：
        #   - 必须在key_is_renamed返回True后调用
        #   - 错误信息包含具体的新键名定位
        #   - 使用KeyError异常类型保持与字典操作的一致性

        new_key = self.__dict__[CfgNode.RENAMED_KEYS][full_key]
        if isinstance(new_key, tuple):
            msg = " Note: " + new_key[1]  # 提取自定义辅助信息
            new_key = new_key[0]
        else:
            msg = ""
        raise KeyError(
            "Key {} was renamed to {}; please update your config.{}".format(
                full_key, new_key, msg
            )
        )

    def is_new_allowed(self):
        return self.__dict__[CfgNode.NEW_ALLOWED]

    @classmethod
    def load_cfg(cls, cfg_file_obj_or_str):
        """
        # 主要作用：作为统一入口加载多种格式的配置文件
        # 重点：
        #   - 支持三种输入形式：
        #     1. YAML格式字符串
        #     2. YAML文件对象
        #     3. 导出cfg属性的Python源文件对象
        #   - 自动分派到具体加载器（YAML/Python）
        #   - 返回标准化CfgNode实例
        # 注意：
        #   - Python源文件必须导出cfg字典/CfgNode属性
        #   - 使用安全YAML解析防止代码注入
        #   - 输入类型错误会立即触发验证异常
        """

        _assert_with_logging(
            isinstance(cfg_file_obj_or_str, _FILE_TYPES + (str,)),
            "Expected first argument to be of type {} or {}, but it was {}".format(
                _FILE_TYPES, str, type(cfg_file_obj_or_str)
            ),
        )
        # 类型分派核心逻辑
        if isinstance(cfg_file_obj_or_str, str):
            return cls._load_cfg_from_yaml_str(
                cfg_file_obj_or_str)  # YAML字符串处理
        elif isinstance(cfg_file_obj_or_str, _FILE_TYPES):
            return cls._load_cfg_from_file(cfg_file_obj_or_str)  # 文件对象处理
        else:
            raise NotImplementedError(
                "Impossible to reach here (unless there's a bug)")

    @classmethod
    def _load_cfg_from_file(cls, file_obj):
        """
        # 主要作用：根据文件扩展名分派具体加载逻辑
        # 重点：
        #   - 自动识别.yaml/.yml和.py后缀文件
        #   - YAML文件直接读取内容并解析
        #   - Python文件通过导入机制获取cfg属性
        #   - 严格限制支持的文件类型扩展名
        # 注意：
        #   - Python文件必须位于可导入路径
        #   - YAML解析使用安全加载方式
        #   - 依赖类常量_YAML_EXTS/_PY_EXTS定义支持格式
        """
        _, file_extension = os.path.splitext(file_obj.name)
        if file_extension in _YAML_EXTS:
            return cls._load_cfg_from_yaml_str(file_obj.read())  # 转交YAML解析器
        # elif file_extension in _PY_EXTS:
        #     return cls._load_cfg_py_source(file_obj.name)  # 转交Python加载器
        else:
            raise Exception(
                "Attempt to load from an unsupported file type {}; "
                "only {} are supported".format(
                    file_obj, _YAML_EXTS.union(_PY_EXTS))
            )

    @classmethod
    def _load_cfg_from_yaml_str(cls, str_obj):
        """Load a config from a YAML string encoding."""
        cfg_as_dict = yaml.safe_load(str_obj)
        return cls(cfg_as_dict)

    # @classmethod
    # def _load_cfg_py_source(cls, filename):
    #     # 主要作用：从Python源文件动态加载配置对象
    #     # 重点：
    #     #   - 动态导入指定文件为Python模块
    #     #   - 强制要求模块必须包含cfg属性
    #     #   - 校验cfg属性的类型有效性（dict/CfgNode）
    #     #   - 通过类构造器实现配置对象安全转换
    #     # 注意：
    #     #   - 使用唯一模块名防止命名冲突
    #     #   - 文件路径需在Python可访问路径中
    #     #   - 依赖_load_module_from_file实现安全导入
    #
    #     """Load a config from a Python source file."""
    #     # 动态导入模块（隔离在独立命名空间中）
    #     module = _load_module_from_file("yacs.config.override", filename)
    #     # 验证模块结构完整性
    #     _assert_with_logging(
    #         hasattr(module, "cfg"),
    #         "Python module from file {} must have 'cfg' attr".format(filename),
    #     )
    #     # 严格类型校验白名单
    #     VALID_ATTR_TYPES = {dict, CfgNode}
    #     _assert_with_logging(
    #         type(module.cfg) in VALID_ATTR_TYPES,
    #         "Imported module 'cfg' attr must be in {} but is {} instead".format(
    #             VALID_ATTR_TYPES, type(module.cfg)
    #         ),
    #     )
    #     # 将原始配置转换为CfgNode实例
    #     return cls(module.cfg)

    @classmethod
    def _decode_cfg_value(cls, value):
        """# 主要作用：将原始配置值转换为合法的Python对象
        # 重点：
        #   - 自动转换字典为CfgNode实例
        #   - 安全解析字符串字面量（使用ast.literal_eval）
        #   - 保留非字符串对象的原始类型
        # 注意：
        #   - 处理YAML解析后可能存在的原始字符串类型
        #   - 捕获ValueError/SyntaxError以保留纯字符串内容
        #   - 避免使用eval()防止代码注入风险"""

        # 处理字典类型：转换为标准CfgNode对象
        if isinstance(value, dict):
            return cls(value)
        # 非字符串类型直接透传
        if not isinstance(value, str):
            return value
        # 安全解析字符串字面量
        try:
            value = literal_eval(value)  # 解析数字/元组/列表等类型
        except (ValueError, SyntaxError):
            # 保留无法解析的纯字符串（如文件路径）
            pass
        return value


load_cfg = (
    CfgNode.load_cfg
)  # keep this function in global scope for backward compatibility


def _valid_type(value, allow_cfg_node=False):
    """
    # 主要作用：验证值是否为配置系统允许的有效类型
    # 重点：
    #   - 使用预定义_VALID_TYPES集合进行基础类型校验
    #   - 通过allow_cfg_node参数控制是否接受CfgNode实例
    #   - 返回布尔值指示类型合法性
    # 注意：
    #   - _VALID_TYPES需包含所有允许的基础类型（如int/str/list等）
    #   - 当allow_cfg_node=True时支持嵌套配置结构
    #   - 该函数是配置类型安全的核心校验器
    """
    return (type(value) in _VALID_TYPES) or (
        allow_cfg_node and isinstance(value, CfgNode)
    )


def _merge_a_into_b(a, b, root, key_list):
    """# 主要作用：递归合并两个配置节点，实现配置覆盖与继承
    # 参数说明：
    #   a: 源配置节点（新配置）
    #   b: 目标配置节点（基础配置）
    #   root: 根配置节点（用于键名检查）
    #   key_list: 当前键路径（用于错误追踪）
    # 重点：
    #   - 深度优先的递归合并策略
    #   - 自动类型转换与校验（_check_and_coerce_cfg_value_type）
    #   - 支持嵌套CfgNode的逐层合并
    #   - 集成弃用/重命名键处理机制
    # 注意：
    #   - a中的键会覆盖b中的同名键
    #   - 合并时会创建配置值的深拷贝
    #   - 依赖根节点的键状态检测方法
    """

    # 前置类型校验（防御性编程）
    _assert_with_logging(
        isinstance(a, CfgNode),
        "`a` (cur type {}) must be an instance of {}".format(type(a), CfgNode),
    )
    _assert_with_logging(
        isinstance(b, CfgNode),
        "`b` (cur type {}) must be an instance of {}".format(type(b), CfgNode),
    )

    for k, v_ in a.items():
        full_key = ".".join(key_list + [k])  # 构造完整键路径用于错误追踪

        # 值预处理流程
        v = copy.deepcopy(v_)  # 防止源数据被意外修改
        v = b._decode_cfg_value(v)  # 解析原始值（如字符串转特定类型）

        # 存在性判断分支
        if k in b:
            # 类型校验与强制转换
            v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)
            # 递归合并嵌套配置
            if isinstance(v, CfgNode):
                try:
                    _merge_a_into_b(v, b[k], root, key_list + [k])  # 深度优先合并
                except BaseException:
                    raise  # 保留原始堆栈信息
            else:
                b[k] = v  # 覆盖基础配置
        elif b.is_new_allowed():
            b[k] = v  # 允许新增配置键
        else:
            # 异常路径处理
            if root.key_is_deprecated(full_key):
                continue  # 静默跳过弃用键
            elif root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)  # 重命名键引导
            else:
                raise KeyError("Non-existent config key: {}".format(full_key))


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """
    # 主要作用：确保配置值替换时的类型兼容性，执行安全类型转换
    # 重点：
    #   - 允许有限类型转换（list <-> tuple，py2下str <-> unicode）
    #   - 精确记录类型不匹配的完整上下文信息
    #   - 作为配置合并时的类型安全屏障
    # 注意：
    #   - 仅当原始类型与目标类型存在明确对应关系时转换
    #   - 使用full_key提供完整的错误定位路径
    #   - 禁止隐式类型转换（如int转float）
    """
    original_type = type(original)
    replacement_type = type(replacement)

    # 基础类型匹配快速返回
    if replacement_type == original_type:
        return replacement

    # 定义允许的类型转换对
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)  # 执行安全转换
        else:
            return False, None

    # 构建转换规则白名单
    casts = [(tuple, list), (list, tuple)]  # 序列类型互转
    # Python 2兼容性处理（自动跳过Python 3环境）
    try:
        casts.append((str, unicode))  # noqa: F821  # py2字符串兼容处理
    except Exception:
        pass

    # 遍历执行允许的类型转换
    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    # 无可转换类型时抛出详细异常
    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def _assert_with_logging(cond, msg):
    """
    # 主要作用：增强版断言，在断言失败时记录日志
    # 重点：
    #   - 结合断言机制与日志记录，方便问题追踪
    #   - 仅在断言失败时记录DEBUG级别日志
    #   - 保持与标准assert语句相同的中断行为
    """
    if not cond:
        logger.debug(msg)
    assert cond, msg

# def _load_module_from_file(name, filename):
#     # 主要作用：跨Python版本实现动态模块加载
#     # 重点：
#     #   - 兼容Python 2和Python 3的导入机制
#     #   - 使用官方推荐的低级导入API
#     #   - 在独立命名空间执行模块代码
#     # 注意：
#     #   - 会实际执行目标模块的顶层代码
#     #   - 需确保文件路径可信（防范代码注入）
#     #   - 推荐用于配置加载等受控场景
#
#     # Python 2实现路径
#     if _PY2:
#         # 使用imp模块直接加载源文件（注意：会执行模块代码）
#         module = imp.load_source(name, filename)
#     # Python 3+实现路径
#     else:
#         # 使用importlib构建规范对象
#         spec = importlib.util.spec_from_file_location(name, filename)
#         # 创建空白模块对象
#         module = importlib.util.module_from_spec(spec)
#         # 执行模块代码（与import语义相同）
#         spec.loader.exec_module(module)
#     return module
