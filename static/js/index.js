window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

})


$(document).querySelectorAll('.slider-container').forEach(container => {
	const bottomVideo = container.querySelector('.bottom-video');
	const topVideo = container.querySelector('.top-video');
	const slider = container.querySelector('.slider');
	let isDragging = false;

	// 更新滑块位置和上层视频裁剪
	function updateClipPath(x) {
	  const rect = container.getBoundingClientRect();
	  const width = rect.width;
	  const clipX = Math.max(0, Math.min(x - rect.left, width));
	  topVideo.style.clipPath = `inset(0 ${width - clipX}px 0 0)`;
	  slider.style.left = `${clipX}px`;
	}

	// 同步视频播放时间
	function syncVideos() {
	  const timeDiff = Math.abs(bottomVideo.currentTime - topVideo.currentTime);
	  if (timeDiff > 0.01) {
		topVideo.currentTime = bottomVideo.currentTime;
	  //   bottomVideo.currentTime = topVideo.currentTime;
	  }
	}

	// 确保两个视频同时播放或暂停
	function syncPlayState() {
	  if (bottomVideo.paused && !topVideo.paused) {
		  bottomVideo.play();
	  } else if (!bottomVideo.paused && topVideo.paused) {
		  bottomVideo.pause();
	  }
	}

	// 定期检查同步
	setInterval(() => {
	  syncVideos();
	  syncPlayState();
	}, 100);

	// 滑块拖动事件
	slider.addEventListener('mousedown', (e) => {
	  isDragging = true;
	  e.preventDefault();
	});

	slider.addEventListener('touchstart', (e) => {
	  isDragging = true;
	  e.preventDefault();
	});

	// 全局鼠标/触摸移动事件，检查当前 container
	document.addEventListener('mousemove', (e) => {
	  if (isDragging) {
		updateClipPath(e.clientX);
	  }
	});

	document.addEventListener('touchmove', (e) => {
	  if (isDragging) {
		updateClipPath(e.touches[0].clientX);
	  }
	});

	// 停止拖动
	document.addEventListener('mouseup', () => {
	  isDragging = false;
	});

	document.addEventListener('touchend', () => {
	  isDragging = false;
	});

	// 视频加载完成后初始化同步
	bottomVideo.addEventListener('loadedmetadata', () => {
	  topVideo.currentTime = bottomVideo.currentTime;
	});

	// 窗口大小变化时更新滑块位置
	window.addEventListener('resize', () => {
	  updateClipPath(container.getBoundingClientRect().left + container.getBoundingClientRect().width / 2);
	});

	// 初始化滑块位置
	updateClipPath(container.getBoundingClientRect().left + container.getBoundingClientRect().width / 2);
  });