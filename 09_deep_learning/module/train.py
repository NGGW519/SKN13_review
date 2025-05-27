
import torch
import time

def test_multi_classification(dataloader, model, loss_fn, device="cpu") -> tuple:
    """
    다중 분류 검증/평가 함수

    Args:
        dataloader: DataLoader - 검증할 대상 데이터로더
        model: 검증할 모델
        loss_fn: 모델 추정값과 정답의 차이를 계산할 loss 함수.
        device: str - 연산을 처리할 장치. default-"cpu", gpu-"cuda"
    Returns:
        tuple: (loss, accuracy)
    """
    model.to(device)                    # 모델을 지정된 장치(cpu/gpu)로 이동
    model.eval()                        # 평가 모드로 설정 (dropout 등 비활성화)
    size = len(dataloader.dataset)      # 전체 데이터 수
    num_steps = len(dataloader)         # 배치 수

    test_loss, test_accuracy = 0., 0.   # 초기화

    with torch.no_grad():               # 그래디언트 계산 비활성화
        for X, y in dataloader:         # 배치 단위 반복
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            ## 정확도 계산
            pred_label = torch.argmax(pred, axis=-1)    # 예측 클래스
            test_accuracy += torch.sum(pred_label == y).item()

        test_loss /= num_steps          # 평균 loss
        test_accuracy /= size           # 전체 대비 정확도

    return test_loss, test_accuracy     # 튜플 반환

def test_binary_classification(dataloader, model, loss_fn, device="cpu") -> tuple:
    """
    이진 분류 검증/평가 함수
    >>>> 다중분류와 거의 동일하되, 정확도 계산방식만 다름

    Args:
        dataloader: DataLoader - 검증할 대상 데이터로더
        model: 검증할 모델
        loss_fn: 모델 추정값과 정답의 차이를 계산할 loss 함수.
        device: str - 연산을 처리할 장치. default-"cpu", gpu-"cuda"
    Returns:
        tuple: (loss, accuracy)
    """
    model.to(device)
    model.eval()                    # 평가 모드로 설정 (dropout 등 비활성화)
    size = len(dataloader.dataset)
    num_steps = len(dataloader)

    test_loss, test_accuracy = 0., 0.

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            ## 정확도 계산
            pred_label = (pred >= 0.5).type(torch.int32)  # 0.5 이상이면 1로 분류
            test_accuracy += (pred_label == y).sum().item() 

        test_loss /= num_steps
        test_accuracy /= size      #전체 개수로 나눈다.
    return test_loss, test_accuracy    

def train(dataloader, model, loss_fn, optimizer, device="cpu", mode:"binary or multi"='binary'):
    """
    모델을 1 epoch 학습시키는 함수

    Args:
        dataloader: DataLoader - 학습데이터셋을 제공하는 DataLoader
        model - 학습대상 모델
        loss_fn: 모델 추정값과 정답의 차이를 계산할 loss 함수.
        optimizer - 최적화 함수
        device: str - 연산을 처리할 장치. default-"cpu", gpu-"cuda"
        mode: str - 분류 종류. binary 또는 multi 

    Returns:
        tuple: 학습후 계산한 Train set에 대한  train_loss, train_accuracy
    """
    model.train()                           # 학습 모드로 전환
    size = len(dataloader.dataset)          # 전체 학습 데이터 수(총 데이터포인트 개수)

    for X, y in dataloader:                 # 배치 단위 반복
        X, y = X.to(device), y.to(device)   # 1. DEVICE로 이동.
        pred = model(X)                     # 2. forward(모델 추정)
        loss = loss_fn(pred, y)             # 3. loss 계산
        optimizer.zero_grad()               # 4. 기존 gradient 초기화
        loss.backward()                     # 5. 역전파로 gradient 계산
        optimizer.step()                    # 6. 파라미터 업데이트

    # 학습이 끝난 후에 동일 train 데이터를 평가    
    if mode == 'binary':
        train_loss, train_accuracy = test_binary_classification(dataloader, model, loss_fn, device)
    else:
        train_loss, train_accuracy = test_multi_classification(dataloader, model, loss_fn, device)
    return train_loss, train_accuracy



def fit(train_loader, val_loader, model, loss_fn, optimizer, epochs, save_best_model=True, 
        save_model_path=None, early_stopping=True, patience=10, device='cpu',  mode:"binary or multi"='binary',
        lr_scheduler=None):
    """
    모델을 학습시키는 함수
    >>>>> 전체 학습 루프(Epoch 단위)

    Args:
        train_loader (Dataloader): Train dataloader
        test_loader (Dataloader): validation dataloader
        model (Module): 학습시킬 모델
        loss_fn (_Loss): Loss function
        optimizer (Optimizer): Optimizer
        epochs (int): epoch수
        save_best_model (bool, optional): 학습도중 성능개선시 모델 저장 여부. Defaults to True.
        save_model_path (str, optional): save_best_model=True일 때 모델저장할 파일 경로. Defaults to None.
        early_stopping (bool, optional): 조기 종료 여부. Defaults to True.
        patience (int, optional): 조기종료 True일 때 종료전에 성능이 개선될지 몇 epoch까지 기다릴지 epoch수. Defaults to 10.
        device (str, optional): device. Defaults to 'cpu'.
        mode(str, optinal): 분류 종류. "binary(default) or multi
        lr_scheduler: Learning Rate Scheduler 객체. default: None, Epoch 단위로 LR 를 변경.

    Returns:
        tuple: 에폭 별 성능 리스트. (train_loss_list, train_accuracy_list, validation_loss_list, validataion_accuracy_list)
    """

    train_loss_list = []                # 결과 저장용 리스트
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []


    if save_best_model:
        best_score_save = torch.inf     # 최고 성능 저장 초기값

    ############################
    # early stopping
    #############################
    if early_stopping:
        trigger_count = 0
        best_score_es = torch.inf       # early stopping 기준 점수

    # 모델 device로 옮기기
    model = model.to(device)          # 모델 장치로 이동
    s = time.time()                   # 시간 측정 시작

    for epoch in range(epochs):       # 에폭 반복
        # 1 epoch 학습
        train_loss, train_accuracy = train(train_loader, model, loss_fn, optimizer, device=device, mode=mode)

        ############ 1 epoch 학습 종료 후-> LR를 조정 (있으면) ########### 
        if lr_scheduler is not None: 
            current_lr = lr_scheduler.get_last_lr()[0]  # log용
            lr_scheduler.step()
            new_lr = lr_scheduler.get_last_lr()[0] # log용
            if current_lr != new_lr: # LR가 변경되었으면
                print(f">>>>>>Learning Rate가 {current_lr}에서 {new_lr}로 변경됨<<<<<<")


        # 검증 데이터셋 평가
        if mode == "binary":
            val_loss, val_accuracy = test_binary_classification(val_loader, model, loss_fn, device=device)
        else:
            val_loss, val_accuracy = test_multi_classification(val_loader, model, loss_fn, device=device)

        # 에폭별 결과 저장 및 출력
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)

        print(f"Epoch[{epoch+1}/{epochs}] - Train loss: {train_loss:.5f} Train Accucracy: {train_accuracy:.5f} || Validation Loss: {val_loss:.5f} Validation Accuracy: {val_accuracy:.5f}")
        print('='*100)

        # 가장 좋은 모델 저장
        if save_best_model:
            if val_loss < best_score_save:
                torch.save(model, save_model_path)
                print(f"저장: {epoch+1} - 이전 : {best_score_save}, 현재: {val_loss}")
                best_score_save = val_loss

        # Early Stopping 처리            
        if early_stopping:
            if val_loss < best_score_es: 
                best_score_es = val_loss  
                trigger_count = 0

            else:
                trigger_count += 1                
                if patience == trigger_count:
                    print(f"Early stopping: Epoch - {epoch}")
                    break

    e = time.time()
    print(e-s, "초")
    return train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list
