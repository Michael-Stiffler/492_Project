var map;
var canvas;
var markers;
var landslide;

function preload(){
    var AJAXtojson = $.ajax({
        type: 'GET',
        url: '/getlandslide',
        async: true,
        success: function(response) { 
            landslide = JSON.parse(JSON.stringify(response));
        },
        contentType: "application/json",
    });

    var AJAXtojson = $.ajax({
        type: 'GET',
        url: '/getlandslide',
        async: false,
        success: function(response) { 
            landslide = JSON.parse(JSON.stringify(response));
        },
        contentType: "application/json",
    });
}

function setup() {
    canvas = createCanvas(1400, 800);
    canvas.parent("map");
    translate(width / 2, height / 2);

    markers = L.markerClusterGroup({
        disableClusteringAtZoom: 6,
        maxClusterRadius: 100,
        animateAddingMarkers: true
    });

    var bounds = L.latLngBounds(L.latLng(-90, -180), L.latLng(90, 180));

    map = L.map('map', {
        preferCanvas: true,
        minZoom: 2,
        zoomControl: false,
        attributionControl: false,
        maxBounds: bounds
    }).setView([0, 0], 1);

    map.invalidateSize();

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    L.easyButton('<span class="button">&FilledSmallSquare;</span>', function insertMarkers(){
        map.addLayer(markers);
    }).addTo(map);

    L.easyButton('<span class="button">&EmptySmallSquare;</span>', function removeMarkers(){
        map.removeLayer(markers);
    }).addTo(map);

    for (var i = 0; i < landslide["data"].length; i++){
        var lat = landslide["data"][i][8];
        var lon = landslide["data"][i][7];

        var date = landslide["data"][i][2];
        var country =  landslide["data"][i][3];
        var type = landslide["data"][i][6];
        var trigger = landslide["data"][i][4];
        var fatalities = landslide["data"][i][1];
        var injuries = landslide["data"][i][5];
        var size = landslide["data"][i][0];

        if(injuries === ""){
            injuries = 0;
        }

        var marker = L.circleMarker(new L.LatLng(lat, lon), {
                color: 'red'
        })
        .bindPopup("Country: " + country +  
              "<br> Type: " + type +
              "<br> Trigger: " + trigger +
              "<br> Fatalities: " + fatalities +
              "<br> Injuries: " + injuries +
              "<br> Size: " + size +
              "<br> Date: " + date +
              "<br> Latitude: " + lat +
              "<br> Longitude: " + lon)
        .openPopup();

        marker.setRadius(9);

        markers.addLayer(marker);
    }
    
    map.addLayer(markers);
}

function draw() {
}





