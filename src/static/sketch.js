var map;
var canvas;
var userMarker;
var markers;
var userLat = 0;
var userLon = 0;
var landslide;

function preload(){
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
        minZoom: 1,
        zoomControl: false,
        attributionControl: false,
        maxBounds: bounds
    }).setView([0, 0], 1);

    map.invalidateSize();
    map.dragging.disable();

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

        if(injuries === NaN){
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

function mouseClicked() {
    map.invalidateSize();
    if(keyIsDown(CONTROL)){

        if(userMarker){
            map.removeLayer(userMarker);
        }

        var point = L.point(mouseX, mouseY);
        xyToLatLong = map.layerPointToLatLng(point);
        userLat = xyToLatLong.lat;
        userLon = xyToLatLong.lng;
        console.log("lat: " + xyToLatLong.lat + " long: " + xyToLatLong.lng);
        runPyScript();
    }
}

function runPyScript(){

    var input = userLat + "#" + userLon;
    var modelResult;

    var AJAXtoFlask = $.ajax({
        type: "POST",
        url: "/datapull",
        async: true,
        data: { data: input },
        success: function(result) {
            console.log(result);
            modelResult = result;
            userMarker = L.marker(new L.LatLng(userLat, userLon), {}).addTo(map)
            .bindPopup("User Selected Location: <br>Lat: " + userLat + "<br>Lon: " + userLon + "<br>Is: " + result)
            .openPopup();

            var AJAXtoFlask = $.ajax({
                type: "POST",
                url: "/sendtosql",
                async: true,
                data: { data: input, result: modelResult},
                success: function(response) {
                    console.log(response + "Added to database!");
                } 
            });
        } 
    });
}





