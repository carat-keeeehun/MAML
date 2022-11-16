import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner, ProtoLearner
from    copy import deepcopy



class Meta(nn.Module):
    """
    Meta Learner
    """ 
    def __init__(self, config, model_arch):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.inner_lr = config.optim.inner_lr
        self.outer_lr = config.optim.outer_lr
        self.n_way = config.training.n_way
        self.k_shot = config.training.k_shot
        self.k_qry = config.training.k_qry
        self.meta_batch_size = config.training.meta_batch_size
        self.inner_update_steps = config.optim.inner_update_steps
        self.test_update_steps = config.optim.test_update_steps

        self.net = Learner(model_arch, config.data.image_channel, config.data.image_size)
        self.outer_optim = optim.Adam(self.net.parameters(), lr=self.outer_lr)

    # def clip_grad_by_norm_(self, grad, max_norm):
    #     """
    #     in-place gradient clipping.
    #     :param grad: list of gradients
    #     :param max_norm: maximum norm allowable
    #     :return:
    #     """

    #     total_norm = 0
    #     counter = 0
    #     for g in grad:
    #         param_norm = g.data.norm(2)
    #         total_norm += param_norm.item() ** 2
    #         counter += 1
    #     total_norm = total_norm ** (1. / 2)

    #     clip_coef = max_norm / (total_norm + 1e-6)
    #     if clip_coef < 1:
    #         for g in grad:
    #             g.data.mul_(clip_coef)

    #     return total_norm/counter

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.inner_update_steps + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.inner_update_steps + 1)]

    
        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            '''
            This is inner optimization loops.
            1. run the i-th task and compute loss for k = 1 ~ K-1
            2. compute grad on theta_pi
            3. update theta_pi
                    theta_pi = theta_pi - train_lr * grad
            '''
            # inner loop
            for k in range(1, self.inner_update_steps):
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                
                grad = torch.autograd.grad(loss, fast_weights)
                
                fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        '''
        This is outer optimization loops.
        After ending of all tasks, sum over all losses on query set across all tasks.
        And then, optimize theta parameters.
        '''
        loss_q = losses_q[-1] / task_num

        self.outer_optim.zero_grad()
        loss_q.backward()
        self.outer_optim.step()

        accs = np.array(corrects) / (querysz * task_num)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.test_update_steps + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.test_update_steps):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) / querysz

        return accs


class ProtoMeta(nn.Module):
    """
    Prototype + Meta Learner
    """ 
    def __init__(self, config):
        """
        :param args:
        """
        super(ProtoMeta, self).__init__()

        self.outer_lr = config.optim.outer_lr
        self.n_way = config.training.n_way
        self.k_shot = config.training.k_shot
        self.k_qry = config.training.k_qry
        self.meta_batch_size = config.training.meta_batch_size
        self.inner_update_steps = config.optim.inner_update_steps
        self.test_update_steps = config.optim.test_update_steps

        self.net = ProtoLearner(config.training.in_channels, config.training.out_channels, config.training.embedding_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.outer_lr)

    def calculate_prototypes(self, embed_spt, y_spt):
        prototypes = []
        embedding_dim = embed_spt.shape[1]

        y_spt_expand = y_spt.view(-1, 1).expand((embed_spt.shape))
        # aggregate samples from the same class. average them to get prototype
        for label in range(len(y_spt[0].unique())):
            proto = torch.mean(embed_spt[y_spt_expand == label].view(-1, embedding_dim), dim=0)
            prototypes.append(proto)

        return torch.stack(prototypes)

    def prototypical_loss(self, prototypes, embed_qry, y_qry):
        distance_matrix = ((embed_qry[:, :, None] - prototypes.t()[None, :, :]) ** 2).sum(1)
        return F.cross_entropy(-distance_matrix, y_qry.squeeze()), distance_matrix

    def calculate_accuracy(self, distance_matrix, y_qry):
        with torch.no_grad():
            pred_label = torch.min(distance_matrix, 1)[1]
            accuracy = torch.sum(pred_label == y_qry) / y_qry.shape[1]
        return accuracy.item()

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [1, setsz, c_, h, w]
        :param y_spt:   [1, setsz]
        :param x_qry:   [1, querysz, c_, h, w]
        :param y_qry:   [1, querysz]
        :return:
        """
        self.net.train()
        
        # get embeddings of support/query images
        embed_spt = self.net(x_spt.squeeze(0))
        embed_qry = self.net(x_qry.squeeze(0))
                
        # calculate prototype and the loss
        prototypes = self.calculate_prototypes(embed_spt, y_spt)
        loss, distance_matrix = self.prototypical_loss(prototypes, embed_qry, y_qry)

        # finetuning only classifier(prototype network) weights in inner loop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        accs = self.calculate_accuracy(distance_matrix, y_qry)

        return accs


    def validation(self, x_spt, y_spt, x_qry, y_qry):
        with torch.no_grad():
            # get embeddings of support/query images
            embed_spt = self.net(x_spt.squeeze(0))
            embed_qry = self.net(x_qry.squeeze(0))
            
            # calculate prototype and the loss
            prototypes = self.calculate_prototypes(embed_spt, y_spt)
            _, distance_matrix = self.prototypical_loss(prototypes, embed_qry, y_qry)
            
            accs = self.calculate_accuracy(distance_matrix, y_qry)
        
        return accs
    
    
    def test(self, x_spt, y_spt, x_qry, y_qry):
        with torch.no_grad():
            # get embeddings of support/query images
            embed_spt = self.net(x_spt.squeeze(0))
            embed_qry = self.net(x_qry.squeeze(0))
            
            # calculate prototype and the loss
            prototypes = self.calculate_prototypes(embed_spt, y_spt)
            _, distance_matrix = self.prototypical_loss(prototypes, embed_qry, y_qry)
            
            accs = self.calculate_accuracy(distance_matrix, y_qry)
        
        return accs


def main():
    pass


if __name__ == '__main__':
    main()
