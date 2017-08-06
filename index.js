

function viz(div, image) {
    var offset = 0.17915;

    var map = L.map(div).setView(
	[55.750475 - offset, 37.621766],
	11
    );

    L.tileLayer('https://vec02.maps.yandex.net/tiles?l=map&x={x}&y={y}&z={z}&scale=2').addTo(map);

    var overlay = L.imageOverlay(
	image,
	[[55.933 - offset, 37.255],
	 [55.555 - offset, 37.947]]
    ).addTo(map);
}


viz('m21', 'i/21.png');
viz('m25', 'i/25.png');
viz('m32', 'i/32.png');
viz('m16', 'i/16.png');
