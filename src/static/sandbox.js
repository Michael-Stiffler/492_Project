var map;
var canvas;
var markers;

function preload(){
    loadedData = loadTable('static/data/Global_Landslide_Catalog_Export_stripped.csv', 'header');
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

    for(let row of loadedData.rows){
        var lat = row.get("latitude");
        var lon = row.get("longitude");

        var date = row.get("date");
        var country =  row.get("country_name");
        var type = row.get("type");
        var trigger = row.get("landslide_trigger");
        var fatalities = row.get("fatality_count");
        var injuries = row.get("injuries");
        var size = row.get("landslide_size");

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





