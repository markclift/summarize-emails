import math


def split_text(text, total_tokens, max_tokens):
    target_chunks = math.ceil(total_tokens/max_tokens) + 1  #Add 1 as a buffer so we don't slightly go over
    total_words = len(text.split())
    target_words_per_chunk = total_words/target_chunks

    paragraphs = text.split('\n')  # splitting the email body into paragraphs

    chunks = []
    current_chunk = []
    current_chunk_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())  # calculating the number of words in the current paragraph

        # if adding this paragraph doesn't exceed the max words limit
        if current_chunk_word_count + paragraph_word_count <= target_words_per_chunk:
            current_chunk.append(paragraph)
            current_chunk_word_count += paragraph_word_count
        else:
            # add current chunk to chunks and start a new one
            chunks.append('\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_chunk_word_count = paragraph_word_count

    # adding the last chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks