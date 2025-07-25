from PIL import Image

def get_caption(model, image_processor, tokenizer, image):
    
    # preprocess the image
    img = image_processor(image, return_tensors="pt")
    # generate the caption (using greedy decoding by default)
    output = model.generate(**img)
    # decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return caption