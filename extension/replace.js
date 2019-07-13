function post(URL, PARAMS)
{
    var temp = document.createElement("form");
    temp.action = URL;
    temp.method = "post";
    temp.style.display = "none";
    for (var x in PARAMS)
    {
        var opt = document.createElement("textarea");
        opt.name = x;
        opt.value = PARAMS[x];
        // alert(opt.name)
        temp.appendChild(opt);
    }
    document.body.appendChild(temp);
    temp.submit();
    return temp;
}

var newContent='<html><head>Request sent. Please stand by...</body></html>';
// document.body.innerHTML
// document.body.innerHTML=newContent


// var reader = new FileReader();

// reader.onerror = errorHandler;
// reader.onloadend = function(e) {
//   console.log(e.target.result);
// };

// replace_url=chrome.runtime.getURL("new_yelppage.html")
// newContent=reader.readAsText(replace_url);

post('http://localhost:5000/', {url:document.URL});
// post('http://localhost:5000/result', {url:document.URL});
// post('http://localhost:5000/send', {url:document.URL});
window.alert('Detecting fake reviews in: '+document.URL)

// $.post('http://localhost:5000/send', {url:'https://www.yelp.com/biz/hanks-cajun-grill-and-oyster-bar-houston?start=120'}, function(data){
    // document.body.innerHTML=data;
// }

document.body.innerHTML=newContent
