var map;
var canvas;
var userMarker;
var markers;
var userLat = 0;
var userLon = 0;

function preload(){
    loadedData = loadTable('static/data/GLC03122015.csv', 'header');
}

function setup() {
    canvas = createCanvas(800, 800);
    translate(width / 2, height / 2);
    canvas.parent("map");

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

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    for(let row of loadedData.rows){
        var lat = row.get("latitude");
        var lon = row.get("longitude");

        var date = row.get("date_");
        var country =  row.get("countrynam");
        var type = row.get("landslide_");
        var trigger = row.get("trigger");
        var fatalities = row.get("fatalities");
        var injuries = row.get("injuries");
        var size = row.get("landslide1");

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
    if(keyIsDown(CONTROL)){

        if(userMarker){
            map.removeLayer(userMarker);
        }

        xyToLatLong = map.layerPointToLatLng(L.point(mouseX, mouseY));
        userLat = xyToLatLong.lat;
        userLon = xyToLatLong.lng;
        console.log("lat: " + xyToLatLong.lat + " long: " + xyToLatLong.lng);

        var callBack = function() {
            userMarker = L.marker(new L.LatLng(xyToLatLong.lat, xyToLatLong.lng), {}).addTo(map);
        };
        
        map.whenReady(callBack);
    }
}

function keyPressed(){
    if(keyCode === CONTROL){
        map.dragging.disable();
    }
}

function keyReleased() {
    if(keyCode === CONTROL){
        map.dragging.enable();
    }
}

function removeMarkers(){
    map.removeLayer(markers);
}

function insertMarkers(){
    map.addLayer(markers);
}

function runPyScript(){

    var input = userLat + "#" + userLon;

    var AJAXtoFlask = $.ajax({
        type: "POST",
        url: "/datapull",
        async: true,
        data: { data: input },
        success: function(result) {
            console.log("Result:");
            console.log(result);
          } 
    });
}





