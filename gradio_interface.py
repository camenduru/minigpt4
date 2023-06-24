
# ========================================
#          Gradio Settings
# ========================================


def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your images first', interactive=False), gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list


def upload_img(gr_imgs, text_input, chat_state, img_names):
    if gr_imgs is None:
        return None, None, gr.update(interactive=True), chat_state, None, img_names

    chat_state = CONV_VISION.copy()
    img_list = []
    for img, name in zip(gr_imgs, img_names):
        try:
            img = preprocess_image(img)
            img_list.append((img, name))
        except Exception as e:
            return handle_error(f"Error preprocessing image '{name}': {str(e)}"), None, gr.update(interactive=True), chat_state, None, img_names

    llm_message = chat.upload_img_batch(img_list, chat_state)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list, img_names


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, img_names, num_beams, temperature):
    answers = chat.answer_batch(conv=chat_state, img_list=img_list, max_new_tokens=300,
                                num_beams=1, temperature=temperature, max_length=2000)
    for i, answer in enumerate(answers):
        chatbot[i][1] = answer

    # Create CSV file using utility function
    csv_filename = create_csv_file(chatbot, img_names)

    return chatbot, chat_state, img_list, img_names, csv_filename

