from django import forms
 
class PictForm(forms.Form):
    image = forms.ImageField(widget=forms.FileInput(attrs={'class':'custom-file-input'}))