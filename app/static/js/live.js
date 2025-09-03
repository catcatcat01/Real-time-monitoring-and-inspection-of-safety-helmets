document.addEventListener('DOMContentLoaded', function() {
    // 连接到Socket.IO
    const socket = io.connect('/live');

    // 启动视频流
    socket.emit('start_stream');

    // 处理检测结果
    socket.on('detection', function(data) {
        updateDetectionResults(data);
    });

    // 连接状态
    socket.on('connect', function() {
        console.log('Connected to WebSocket');
    });

    socket.on('disconnect', function() {
        console.log('Disconnected from WebSocket');
    });

    function updateDetectionResults(data) {
        const container = document.getElementById('resultsContainer');
        const item = document.createElement('div');
        item.className = `result-item ${data.has_unauthorized ? 'unauthorized' : 'authorized'}`;

        let html = `<strong>时间:</strong> ${data.time_str}<br>`;
        if (data.has_unauthorized) {
            html += `<span class="text-danger">未佩戴安全帽: ${data.unauthorized_count}</span><br>`;
            if (data.faces && data.faces.length > 0) {
                data.faces.forEach(face => {
                    html += `<div class="mt-2">
                        <strong>${face.name}</strong> (${face.code})<br>
                        <small>相似度: ${(face.similarity * 100).toFixed(1)}%</small>
                    </div>`;
                });
            }
        } else {
            html += `<span class="text-success">全部佩戴安全帽</span>`;
        }

        item.innerHTML = html;
        container.insertBefore(item, container.firstChild);

        // 限制最多显示50条记录
        if (container.children.length > 50) {
            container.removeChild(container.lastChild);
        }
    }
});