import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def do_predict(word_to_id, model, string):
    """
    입력에 대한 답변 생성하는 함수
    :param word_to_id: vocabulary
    :param model: model
    :param string: 입력 문자열
    """
    # token 생성
    token = [word_to_id[w] for w in string.strip().split()]

    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([token]).to(args.device)
        logits = model(inputs)
        _, indices = logits.max(-1)
        y_pred = indices[0].numpy()
    result = "학생" if y_pred == 1 else "기타"
    return result


def draw_history(history):
    """
    학습과정 그래프 출력
    :param history: 학습 이력
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], "b-", label="train_loss")
    plt.plot(history["valid_loss"], "r--", label="valid_loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], "k--", label="train_acc")
    plt.plot(history["valid_acc"], "k--", label="valid_acc")
    plt.xlabel("Epoch")
    plt.legend()

    plt.show()


def accuracy_fn(logits, labels):
    """
    model accuracy 측정
    :param logits: 예측 값
    :param labels: 정답
    """
    # 값이 최대인 index 값
    _, indices = logits.max(-1)
    # label과 비교
    matchs = torch.eq(indices, labels).cpu().numpy()
    total = np.ones_like(matchs)
    acc_val = np.sum(matchs) / max(1, np.sum(total))
    return acc_val


def eval_epoch(args, model, loader, loss_fn):
    """
    1 epoch 평가
    :param args: 입력 arguments
    :param model: 모델
    :param loader: 데이터로더
    :param loss_fn: loss 계산함수
    """
    # model을 eval 모드로 전환
    model.eval()
    # loss 및 accuracy 저장
    losses, access = [], []
    # 실행시에 gradint 계산 비활성화
    with torch.no_grad():
        for batch in loader:
            # batch 입력값 처리 (CPU or GPU)
            inputs, labels = map(lambda v: v.to(args.device), batch)
            # 모델 실행
            logits = model(inputs)
            # loss 계산
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            # loss 저장
            loss_val = loss.item()
            losses.append(loss_val)
            # accuracy 계산 및 저장
            acc_val = accuracy_fn(logits, labels)
            access.append(acc_val)

    return np.mean(losses), np.mean(access)


def train_epoch(args, model, loader, loss_fn, optimizer):
    """
    1 epoch 학습
    :param args: 입력 arguments
    :param model: 모델
    :param loader: 데이터로더
    :param loss_fn: loss 계산함수
    :param optimizer: optimizer
    """
    # model을 train 모드로 전환
    model.train()
    # loss 및 accuracy 저장
    losses, access = [], []
    # data loader에서 batch단위로 처리
    for batch in loader:
        # optimizer 초기화
        optimizer.zero_grad()
        # batch 입력값 처리 (CPU or GPU)
        inputs, labels = map(lambda v: v.to(args.device), batch)
        # 모델 실행
        logits = model(inputs)
        # loss 계산
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        # model weight 변경
        optimizer.step()
        # loss 저장
        loss_val = loss.item()
        losses.append(loss_val)
        # accuracy 계산 및 저장
        acc_val = accuracy_fn(logits, labels)
        access.append(acc_val)

    return np.mean(losses), np.mean(access)


class SentencePrediction(torch.nn.Module):
    """ 문장단위 예측 모델 """

    def __init__(self, n_vocab):
        """
        생성자
        :param n_vocab: number of vocab
        """
        super().__init__()
        self.embed = torch.nn.Embedding(n_vocab, 4)
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, inputs):
        """
        모델 실행
        :param inputs: input data
        """
        hidden = self.embed(inputs)
        hidden, _ = torch.max(hidden, dim=1)
        logits = self.linear(hidden)
        return logits


class SimpleDataSet(torch.utils.data.Dataset):
    """ 데이터셋 클래스 """

    def __init__(self, inputs, labels):
        """
        생성자
        :param inputs: 입력
        :param labels: 정답
        """
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        """ 데이터셋 길이 """
        assert len(self.inputs) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, index):
        """
        데이터 한 개 조회
        :param index: 데이터 위치
        """
        return (
            torch.tensor(self.inputs[index]),
            torch.tensor(self.labels[index]),
        )

    def collate_fn(self, batch):
        """
        batch단위로 데이터 처리
        :param batch: batch 단위 데이터
        """
        inputs, labels = list(zip(*batch))

        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        labels = torch.stack(labels)

        batch = [
            inputs,
            labels,
        ]

        return batch


def make_data(args, word_to_id, raw_inputs, raw_labels, shuffle=False):
    """
    학습용 데이터 생성
    :param args: 입력 arguments
    :param word_to_id: vocabulary
    :param raw_inputs: 입력 문장
    :param raw_labels: 정답
    :param shuffle: 데이터 순서 shuffle 여부
    """
    # 입력 데이터
    inputs = []
    for s in raw_inputs:
        inputs.append([word_to_id[w] for w in s.split()])
    # 정답 데이터
    labels = raw_labels
    # dataset
    dataset = SimpleDataSet(inputs, labels)
    # random sample data
    sampler = torch.utils.data.RandomSampler(dataset) if shuffle else None
    # data loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.n_batch, sampler=sampler, collate_fn=dataset.collate_fn)

    return loader


def make_vocab(args, raw_inputs):
    """
    vocabulary 정의
    :param args: 입력 arguments
    :param sentences: 입력 문장
    """
    # 문장을 띄어쓰기단위로 분할
    words = []
    for s in raw_inputs:
        words.extend(s.split())
    # 중복단어 제거
    words = list(dict.fromkeys(words))
    # 각 단어별 일련번호
    word_to_id = {"[PAD]": 0, "[UNK]": 1}
    for w in words:
        word_to_id[w] = len(word_to_id)
    # 각 번호별 단어
    id_to_word = {i: w for w, i in word_to_id.items()}

    return word_to_id, id_to_word


def raw_data(args):
    """
    학습에 필요한 데이터 정의
    :param args: 입력 arguments
    """
    # 입력 문장
    raw_inputs = ["나는 학생 입니다", "나는 좋은 선생님 입니다", "당신은 매우 좋은 선생님 입니다"]
    # 정답: 학생(1), 기타(0)
    raw_labels = [1, 0, 0]

    return raw_inputs, raw_labels


def main(args):
    """
    동작을 실행하는 main 함수
    :param args: 입력 arguments
    """
    # 데이터 준비
    raw_inputs, raw_labels = raw_data(args)
    # vocabulary 생성
    word_to_id, id_to_word = make_vocab(args, raw_inputs)

    #
    # 학습 과정
    #

    # 학습용 데이터 생성
    train_loader = make_data(args, word_to_id, raw_inputs, raw_labels, shuffle=True)
    valid_loader = make_data(args, word_to_id, raw_inputs, raw_labels)
    # 학습용 모델 생성
    model = SentencePrediction(len(word_to_id))
    model.to(args.device)
    # loss & optimizer 생성
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 학습 history
    history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
    # 가장 좋은 acc 값
    best_acc = 0
    # 학습 및 평가
    for e in range(args.n_epoch):
        train_loss, train_acc = train_epoch(args, model, train_loader, loss_fn, optimizer)
        valid_loss, valid_acc = eval_epoch(args, model, valid_loader, loss_fn)
        # 학습 history 저장
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)
        # 학습과정 출력
        print(f"eopch: {e + 1:3d}, train_loss: {train_loss:.5f}, train_acc: {train_acc: .5f}, valid_loss: {valid_loss:.5f}, valid_acc: {valid_acc:.5f}")
        # best weight 저장
        if best_acc < valid_acc:
            best_acc = valid_acc
            # 저장
            torch.save(
                {"state_dict": model.state_dict(), "valid_acc": valid_acc},
                args.save_path,
            )
            # 저장내용 출력
            print(f"  >> save weights: {args.save_path}")

    draw_history(history)

    #
    # 검증과정
    #

    # 검증용 데이터 생성
    test_loader = make_data(args, word_to_id, raw_inputs, raw_labels)
    # 검증용 모델 생성
    model = SentencePrediction(len(word_to_id))
    model.to(args.device)
    # 저장된 weight load
    save_dict = torch.load(args.save_path)
    model.load_state_dict(save_dict["state_dict"])
    # 평가
    test_loss, test_acc = eval_epoch(args, model, test_loader, loss_fn)
    print()
    print(f"test_loss: {test_loss:.5f}, test_acc: {test_acc:.5f}")

    #
    # 배포
    #
    # 배치용 모델 생성
    model = SentencePrediction(len(word_to_id))
    model.to(args.device)
    # 저장된 weight load
    save_dict = torch.load(args.save_path)
    model.load_state_dict(save_dict["state_dict"])
    # 예측 실행
    result = do_predict(word_to_id, model, "당신은 선생님 입니다")
    print()
    print(result)


def set_seed(seed):
    """ random seed 설정 """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    """ 동작에 필요한 arguments 설정 """
    parser = argparse.ArgumentParser(description="Sentence prediction simple project arguments.")

    parser.add_argument("--seed", default=1234, type=int, help="random seed value")
    parser.add_argument("--n_epoch", default=30, type=int, help="number of epoch")
    parser.add_argument("--n_batch", default=2, type=int, help="number of batch")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument(
        "--save_path",
        default="01-02-sentence-prediction.pth",
        type=str,
        help="save weights path",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    # CPU 또는 GPU 사용여부 결정
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
