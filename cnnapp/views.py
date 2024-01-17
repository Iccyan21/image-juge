from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.template import loader
from .forms import PictForm
from .models import Pict
 
def index(request):
    template = loader.get_template('cnnapp/index.html')
    context = {'form': PictForm()}
    return HttpResponse(template.render(context, request))
 
def predict(request):
    if not request.method == 'POST':
        return
        redirect('cnnapp:index')
 
    form = PictForm(request.POST, request.FILES)
    if not form.is_valid():
        raise ValueError('Form is illegal.')
 
    pict = Pict(image=form.cleaned_data['image'])
    predicted, rate = pict.predict()
 
    template = loader.get_template('cnnapp/result.html')
 
    context = {
        'pict_name': pict.image.name,
        'pict_data': pict.image_src(),
        'predicted': predicted,
        'rate': rate,
    }
 
    return HttpResponse(template.render(context, request))