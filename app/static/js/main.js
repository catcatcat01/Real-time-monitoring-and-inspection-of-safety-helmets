/**
 * 全局交互逻辑
 */
$(document).ready(function() {
    // 页面加载动画
    $(window).on('load', function() {
        const loader = $('.page-loader');
        if (loader.length) {
            loader.fadeOut(500);
        }
    });

    // 平滑滚动
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // 处理文件选择预览
    $('#videoFile').change(function(e) {
        if (e.target.files.length > 0) {
            const fileName = e.target.files[0].name;
            const fileSize = (e.target.files[0].size / (1024 * 1024)).toFixed(2);
            $(this).next('.form-text').text(`已选择: ${fileName} (${fileSize} MB)`);
        }
    });
});

/**
 * 工具函数
 */
const AppUtils = {
    // 格式化时间
    formatTime(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        return [
            h.toString().padStart(2, '0'),
            m.toString().padStart(2, '0'),
            s.toString().padStart(2, '0')
        ].join(':');
    },

    // 显示通知
    showNotification(message, type = 'info') {
        const notification = $(`
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `);
        $('main').prepend(notification);

        // 3秒后自动关闭
        setTimeout(() => {
            notification.alert('close');
        }, 3000);
    }
};