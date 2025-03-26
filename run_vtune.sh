#!/bin/bash
#SBATCH -n 1
#SBATCH -o slurm-output/vtune-job-%j.out
#SBATCH -e slurm-error/vtune-job-%j.err
#SBATCH -c 64
#SBATCH --exclusive
#SBATCH --exclude hepnode0

# 使用说明：
# sbatch ./runvtune.sh <version> <test_set>
# 参数1 <version>: 版本号，例如 "v0.3.2"
# 参数2 <test_set>: 测试集类型，0=small, 1=vgg16

# 默认值设置
DEFAULT_VERSION="default"
DEFAULT_TEST="0"
DEFAULT_PORT="8080"

# 检查版本参数
if [ -z "$1" ]; then
    VERSION="$DEFAULT_VERSION"
    echo "未指定版本号，使用默认版本: $DEFAULT_VERSION"
else
    VERSION="$1"
    echo "使用版本: $VERSION"
fi

# 检查测试集参数
if [ "$2" == "1" ]; then
    CONFIG_FILE="conf/vgg16.conf"
    TEST_NAME="vgg16"
    TAG="-vgg16"
    echo "使用大测试集 (VGG16)"
else
    CONFIG_FILE="conf/small.conf"
    TEST_NAME="small"
    TAG="-small"
    echo "使用小测试集 (Small)"
fi

# 检查端口参数（可选）
if [ -z "$3" ]; then
    PORT="$DEFAULT_PORT"
else
    PORT="$3"
fi

# 创建基于版本的结果目录
BASE_RESULT_DIR="vtune_results/$VERSION-$TEST_NAME"
mkdir -p "$BASE_RESULT_DIR"

# 加载 VTune 环境
eval $(spack load --sh vtune)

# 检查 VTune 是否成功加载
if ! command -v vtune &> /dev/null; then
    echo "错误: VTune 未能成功加载，尝试其他加载方法..."
    
    # 尝试其他可能的加载方法
    if [ -f "/opt/intel/oneapi/vtune/latest/vtune-vars.sh" ]; then
        source /opt/intel/oneapi/vtune/latest/vtune-vars.sh
    elif [ -f "/opt/intel/vtune/vtune-vars.sh" ]; then
        source /opt/intel/vtune/vtune-vars.sh
    fi
    
    # 再次检查
    if ! command -v vtune &> /dev/null; then
        echo "错误: 无法加载 VTune。请确保 VTune 已正确安装。"
        exit 1
    fi
fi

echo "VTune 版本信息："
vtune -version

# 设置 OpenMP 参数（根据你的优化记录）
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# 定义要运行的分析类型
ANALYSIS_TYPES=("hotspots" "memory-access" "threading" "uarch-exploration")

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_LOG="$BASE_RESULT_DIR/analysis_log_${TIMESTAMP}.txt"

{
    echo "==============================================="
    echo " VTune 分析报告 - 版本: $VERSION - 测试集: $TEST_NAME"
    echo " 开始时间: $(date)"
    echo "==============================================="
    
    # 运行所有分析类型
    for ANALYSIS_TYPE in "${ANALYSIS_TYPES[@]}"; do
        echo -e "\n\n"
        echo "--- 开始 $ANALYSIS_TYPE 分析 $(date) ---"
        
        # 创建特定分析类型的子目录
        RESULT_DIR="$BASE_RESULT_DIR/${ANALYSIS_TYPE}_${TEST_NAME}_${TIMESTAMP}"
        mkdir -p "$RESULT_DIR"
        
        echo "收集数据中，请等待..."
        
        # 使用 numactl 绑定到单个 NUMA 节点运行 VTune 分析
        vtune -collect $ANALYSIS_TYPE -result-dir "$RESULT_DIR" -- numactl --cpunodebind=0 --membind=0 ./winograd $CONFIG_FILE
        
        # 检查分析是否成功
        if [ $? -eq 0 ]; then
            echo "数据收集成功！生成报告..."
            
            # 生成文本摘要报告
            vtune -report summary -result-dir "$RESULT_DIR" > "$RESULT_DIR/summary.txt"
            
            # 生成 HTML 报告
            vtune -report $ANALYSIS_TYPE -result-dir "$RESULT_DIR" -report-output "$RESULT_DIR/report.html"
            
            echo "分析完成: $ANALYSIS_TYPE"
            echo "结果保存在: $RESULT_DIR"
        else
            echo "错误: $ANALYSIS_TYPE 分析失败"
        fi
    done
    
    echo -e "\n\n"
    echo "==============================================="
    echo " 所有分析完成！"
    echo " 结束时间: $(date)"
    echo " 结果位置: $BASE_RESULT_DIR"
    echo "==============================================="
} | tee "$RESULT_LOG"

# 创建索引HTML文件
cat > "$BASE_RESULT_DIR/index.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>VTune 分析结果 - 版本 $VERSION</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #0071c5; }
        .analysis-group { margin: 20px 0; }
        .analysis { margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 5px; }
        a { color: #0071c5; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>VTune 分析结果 - 版本 $VERSION</h1>
    <p>测试集: $TEST_NAME</p>
    <p>分析时间: $TIMESTAMP</p>
    
    <h2>分析结果:</h2>
EOF

# 获取所有结果目录并按时间排序
for ANALYSIS_TYPE in "${ANALYSIS_TYPES[@]}"; do
    RESULT_DIRS=$(find "$BASE_RESULT_DIR" -type d -name "${ANALYSIS_TYPE}_${TEST_NAME}_*" | sort -r)
    
    if [ -n "$RESULT_DIRS" ]; then
        cat >> "$BASE_RESULT_DIR/index.html" << EOF
    <div class="analysis-group">
        <h3>$ANALYSIS_TYPE 分析</h3>
EOF
        
        for DIR in $RESULT_DIRS; do
            DIR_NAME=$(basename "$DIR")
            if [ -f "$DIR/report.html" ]; then
                cat >> "$BASE_RESULT_DIR/index.html" << EOF
        <div class="analysis">
            <p><strong>$DIR_NAME</strong></p>
            <p><a href="$(basename $DIR)/report.html">查看详细报告</a> | 
               <a href="$(basename $DIR)/summary.txt">查看摘要</a></p>
        </div>
EOF
            fi
        done
        
        cat >> "$BASE_RESULT_DIR/index.html" << EOF
    </div>
EOF
    fi
done

cat >> "$BASE_RESULT_DIR/index.html" << EOF
    
    <h2>启动 VTune GUI 访问结果</h2>
    <p>在本地机器上执行:</p>
    <pre>vtune-gui --open-result ssh://username@hostname$BASE_RESULT_DIR/&lt;结果目录&gt;</pre>
    
    <h2>使用 VTune 后端服务</h2>
    <p>已在端口 $PORT 启动 VTune 后端服务。</p>
    <p>在本地浏览器中访问: http://hostname:$PORT</p>
</body>
</html>
EOF

echo "创建了索引文件: $BASE_RESULT_DIR/index.html"

# 启动 VTune 后端服务
echo "正在启动 VTune 后端服务，端口: $PORT..."
vtune-backend --data-directory $(dirname $(realpath "$BASE_RESULT_DIR")) --port $PORT --allow-remote-access &
BACKEND_PID=$!

# 记录后端进程ID，便于稍后清理
echo $BACKEND_PID > "$BASE_RESULT_DIR/backend_pid.txt"
echo "VTune 后端服务已启动，PID: $BACKEND_PID"
echo "可通过浏览器访问: http://$(hostname):$PORT"

# 保持后端服务运行一段时间（例如12小时）
echo "后端服务将持续运行12小时，之后自动停止"
sleep 43200 && kill $BACKEND_PID &

# 结束脚本
echo "VTune 分析完成。可使用以下命令停止后端服务:"
echo "kill \$(cat $BASE_RESULT_DIR/backend_pid.txt)"
