$(function(){

    $("#cameraInput").on("change",function(event){
        if(event.target.files.length == 1 && 
           event.target.files[0].type.indexOf("image/") == 0) {
            file = event.target.files[0];
            $("#dropbox").removeClass("emptyDropbox");
        }
    });
	
	var dropbox = $('#dropbox'),
		message = $('.message', dropbox);
	
	dropbox.filedrop({
		paramname: 'file',
		maxfiles: 1,
    	maxfilesize: 10,
		url: '/upload',
		uploadFinished:function(i,file,response){
			$.data(file).addClass('done');
		},
		
    	error: function(err, file) {
			switch(err) {
				case 'BrowserNotSupported':
					showMessage('Your browser does not support HTML5 file uploads!');
					break;
				case 'TooManyFiles':
					alert('Too many files! Please select ' + this.maxfiles + ' at most!');
					break;
				case 'FileTooLarge':
					alert(file.name + ' is too large! The size is limited to ' + this.maxfilesize + 'MB.');
					break;
				default:
					break;
			}
		},
		
		beforeEach: function(file){
			if(!file.type.match(/^image\//)){
				alert('Only images are allowed!');
				return false;
			}
		},
		
		uploadStarted:function(i, file, len){
			createImage(file);
		},
		
		progressUpdated: function(i, file, progress) {
			$.data(file).find('.progress').width(progress);
		}
    	 
	});
	
	var template = '<div id="copy-move-vis" class="vis-wrap">'+
                        '<img />'+
					'</div>'+
                    '<button type="button" class="yesButton" onclick="fetchData()">Process Image</button>'+
                    '<button type="button" class="noButton" onclick="reset()">Reset</button>'
                    ; 
	
	function createImage(file){

		var preview = $(template), 
			image = $('img', preview);
			
		var reader = new FileReader();
		
        image.width = 100;
		image.height = 100;

		reader.onload = function(e){
			image.attr('src',e.target.result);
		};
		
		reader.readAsDataURL(file);
		
		message.hide();
		preview.appendTo(dropbox);
		
		$.data(file,preview);
	}

	function showMessage(msg){
		message.html(msg);
	}

});
