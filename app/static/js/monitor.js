
$(document).ready(function() {
    // 1. 初始化SSE连接，拉取实时流
    let eventSource;
    function connectSSE() {
        // 关闭现有连接（避免重复连接）
        if (eventSource) {
            eventSource.close();
        }

        // 重置状态
        $('#monitorFrame').attr('src', "{{ url_for('static', filename='images/loading.gif') }}");
        $('#currentTime').text('-');
        $('#unauthorizedCount').text('0').removeClass('alert-badge');
        $('#authorizedCount').text('0');
        $('#statUnauthorized').text('0');
        $('#statAuthorized').text('0');
        $('#statPeople').text('无');
        $('#alertContainer').addClass('d-none');

        // 建立SSE连接
        eventSource = new EventSource("{{ url_for('detection.stream_monitor') }}");

        // 处理SSE消息
        eventSource.onmessage = function(e) {
            const data = JSON.parse(e.data);
            switch(data.type) {
                case 'frame':
                    // 更新实时画面
                    $('#monitorFrame').attr('src', data.frameData);
                    // 更新时间和统计
                    $('#currentTime').text(data.timeStr);
                    $('#unauthorizedCount').text(data.unauthorizedCount);
                    $('#authorizedCount').text(data.authorizedCount);
                    $('#statUnauthorized').text(data.unauthorizedCount);
                    $('#statAuthorized').text(data.authorizedCount);
                    // 未佩戴时闪烁提醒
                    if (data.unauthorizedCount > 0) {
                        $('#unauthorizedCount').addClass('alert-badge');
                    } else {
                        $('#unauthorizedCount').removeClass('alert-badge');
                    }
                    // 更新人员信息
                    const faceNames = data.faces.map(face => face.name).filter(name => name !== '未知人员');
                    $('#statPeople').text(faceNames.length > 0 ? faceNames.join(', ') : '无');
                    break;
                case 'error':
                    // 显示错误提示
                    $('#alertContainer').removeClass('d-none alert-warning').addClass('alert-danger');
                    $('#alertContainer').text(`错误: ${data.message}`);
                    break;
                case 'warning':
                    // 显示警告提示（如流中断）
                    $('#alertContainer').removeClass('d-none alert-danger').addClass('alert-warning');
                    $('#alertContainer').text(`警告: ${data.message}`);
                    break;
                case 'end':
                    // 显示结束提示
                    $('#alertContainer').removeClass('d-none alert-danger').addClass('alert-info');
                    $('#alertContainer').text(data.message);
                    break;
            }
        };

        // SSE连接错误处理
        eventSource.onerror = function() {
            $('#alertContainer').removeClass('d-none alert-warning').addClass('alert-danger');
            $('#alertContainer').text('监控连接异常，正在重试...');
            // 3秒后重试
            setTimeout(connectSSE, 3000);
        };
    }

    // 2. 初始化连接
    connectSSE();

    // 3. 刷新监控按钮事件
    $('#refreshStream').click(function() {
        connectSSE();
        $(this).text('刷新中...').prop('disabled', true);
        setTimeout(() => {
            $(this).text('').append('<i class="bi bi-arrow-repeat"></i> 刷新监控').prop('disabled', false);
        }, 1500);
    });

    // 4. 页面关闭时关闭SSE连接
    window.addEventListener('beforeunload', function() {
        if (eventSource) {
            eventSource.close();
        }
    });
});
