import torch


def train(teacher_model, student_model, optimizer, criterion, train_loader, device, mode):
    total_loss = 0
    c_pred = 0
    c_total = 0
    if mode == "LoRA":
        teacher_model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input, mask, target = batch
            input, mask, target = input.to(device), mask.to(device), target.to(device)
            output = teacher_model(input, mask)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            c_pred += torch.sum(pred == target).item()
            c_total += len(target)
    
    elif mode == "distil":
        teacher_model.eval()
        student_model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input, mask, target = batch
            input, mask, target = input.to(device), mask.to(device), target.to(device)
            with torch.no_grad():
                teacher_output = teacher_model(input, mask)
            student_output = student_model(input, mask)
            T = 2
            soft_target = torch.nn.functional.softmax(teacher_output/T, dim=-1)
            log_s= torch.nn.functional.log_softmax(student_output/T, dim=-1)
            loss = torch.nn.functional.kl_div(log_s, soft_target, reduction='batchmean') * T * T
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = torch.argmax(student_output, dim=-1)
            c_pred += torch.sum(pred == target).item()
            c_total += len(target)


    elif mode == "rnn":
        student_model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input, mask, target = batch
            input, mask, target = input.to(device), mask.to(device), target.to(device)
            output = student_model(input, mask)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = torch.argmax(output, dim=-1)
            c_pred += torch.sum(pred == target).item()
            c_total += len(target)


    return total_loss/len(train_loader), c_pred / c_total
    
    

def evaluate(model, criterion, data_loader, device):
    total_loss = 0
    c_pred = 0
    c_total = 0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input, mask, target = batch
            input, mask, target = input.to(device), mask.to(device), target.to(device)
            output = model(input, mask)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            c_pred += torch.sum(pred == target).item()
            c_total += len(target)

    return total_loss/len(data_loader), c_pred / c_total
    
