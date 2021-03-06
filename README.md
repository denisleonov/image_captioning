# image_captioning
HSE deep learning project

Основная суть нашего метода следующая. Наша модель состоит из двух частей -- энкодер и декодер. Энкодер отвечает за получение эмбеддинга картинки, это можно сделать с помощью предобученной сверточной сети, а декодер -- за преобразование эмбеддинга изображения в последовательность токенов слов.

В виде декодера в качестве бейзлайна мы взяли ячейку LSTM и на вход ей подавали эмбеддинг картинки, который по размеру совпадает с эмбеддингом токенов.

## Наши текущие результаты

### `DetectionToLSTM.ipynb`

Здесь в качестве декодера взята сеть `FasteRCNN`, в которой в качестве `backbone` модели используется `ResNet-50-FPN`, эмбеддинги получаются с помощью применения полносвязного слоя к последнему сверточному слою. Все это обучалось на датасете MS COCO. В конце ноутбука можно посмотреть предсказания для подписей в процессе обучения.

### `SimpleCnnToLSTM.ipynb`

В сетке DenseNet121 (она выбрана из-за своего небольшого веса, может еще поменяется) убран слой классификатора и заменен на обычный линейный слой, который и будет выдавать эмбеддинги. Декодер - LSTM. Пока что запускалось только локально, поэтому cверточные слои DenseNet121 были заморожены и обучался только последний линейный слой. Обучение происходило на датасете Flickr8k.

Что планируется сделать в этой части в ближайшее время: разморозить все слои в DenseNet121 (как показала последняя домашняя работа по DL, обучать только последний линейный слой это не очент хорошая идея), посмотреть на то, какие подписи предсказывает модель (пока на обучении смотрелся только  loss). Еще планируется конечно дообучать модель=)

Основной моделью для декодера у нас будет трансформер -- на данный момент `GPT2`.

### `gpt2-resnet-full.ipynb` (новая версия), `gpt2-resnet-draft.ipynb` (старая версия)
Датасет - Google's Conceptual Captions (описание и url картинки).

В качестве энкодера используется `resnet18` (замороженная) + обучаемый слой для проекции в необходимый размер эмбеддингов, в качестве декодера используется `GPT2`. В текущей итерации на вход gpt2 подается последовательность: embed(<IMG>), img_embed, embed(<DESC>), desc_embed_1, desc_embed_2..., где embed - функция взятия эмбеддингов, img_embed - эмбеддинг для картинки (пока что просто один вектор), desc_embed_i - эмбеддинг токена текса, <IMG>, <DESC> - специальные токены. Таким образом, gpt2 подается префикс из двух специальных токенов и одного вектора эмбеддингов картинки. В дальнейшем будут проводиться эксперименты с разными эмбеддингами и специальными токенами, добавлю token_types_ids, которые, возможно, улучшат обучение. 

Помимо архитектуры еще нужно поработать над датасетом, так как он очень медленно работает (грузить картинки из сети довольно долго и проблематично), возможно имеет смысл скачать хотя бы какую-то долю датасета и использовать его, а не все 3 млн примеров.

## Планы на будущее

Планируем экспериментировать с эмбеддингами, есть несколько вариантов:
* для начала попробуем использовать несколько эмбеддингов с разных слоев
* разобьем картинку на несколько частей и с каждой возьмем эмбеддинги
* будем использовать эмбеддинги из object detection моделей

Ожидается,что так модель сможет лучше выделять контекст из изображения, в этом ей поможет attention, который будет брать информацию не только с первого и единственного эмбеддинга картинки, а сразу с нескольких разных эмбеддингов.
