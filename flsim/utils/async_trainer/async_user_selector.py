#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import abc
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np
from flsim.data.data_provider import IFLDataProvider, IFLUserData
from flsim.interfaces.model import IFLModel
from typing import Optional, List, Any
import torch.nn as nn

@dataclass
class AsyncUserSelectorInfo:
    r"""
    Dataclass to encapsulate a selected user for async training

    user_data (IFLUserData): seleected user data in the dataset
    user_index (int): the index for user_data assuming IFLDataProvider.train_users is a List
    """

    user_data: IFLUserData
    user_index: int


class AsyncUserSelector(abc.ABC):
    def __init__(self, data_provider: IFLDataProvider):
        self.data_provider: IFLDataProvider = data_provider

    @abc.abstractmethod
    def get_random_user(self) -> AsyncUserSelectorInfo:
        r"""
        Returns a random IFLUserData from the dataset and the user index (for testing)
        """
        pass


class RandomAsyncUserSelector(AsyncUserSelector):
    def __init__(self, data_provider: IFLDataProvider):
        super().__init__(data_provider)

    def get_random_user(self) -> AsyncUserSelectorInfo:
        user_index = np.random.randint(0, self.data_provider.num_train_users())
        return AsyncUserSelectorInfo(
            user_data=self.data_provider.get_train_user(user_index),
            user_index=user_index,
        )

class CustomAsyncUserSelector(AsyncUserSelector):
    def __init__(self, data_provider: IFLDataProvider):
        super().__init__(data_provider)

    def get_random_user(self, all_client_latest_deltas:Optional[List[Any]]=None) -> AsyncUserSelectorInfo:
        assert all_client_latest_deltas is not None
        assert len(all_client_latest_deltas) == self.data_provider.num_train_users()
        user_index = self.get_user_index_with_largest_updates(all_client_latest_deltas)
        print(f'--------------Selected client {user_index}------------')
        # user_index = np.random.randint(0, self.data_provider.num_train_users())
        return AsyncUserSelectorInfo(
            user_data=self.data_provider.get_train_user(user_index),
            user_index=user_index,
        )

    def get_user_index_with_largest_updates(self, all_client_latest_deltas:List[Any]):
        r"""
        all_client_latest_deltas will be a list
        Elements will either be None (never trained before, so no update available)
        or of type nn.Module()
        """
        # Check if we should return a random index (10% chance)
        if np.random.random() < 0.1:
            return np.random.randint(0, len(all_client_latest_deltas))
        
        return_index = -1
        highest_magnitude = -2
        
        # Calculate magnitudes of nn.Module elements and track the one with highest magnitude
        for index, delta in enumerate(all_client_latest_deltas):
            if isinstance(delta, nn.Module):
                magnitude = sum(param.norm().item() for param in delta.parameters())
                if magnitude > highest_magnitude:
                    highest_magnitude = magnitude
                    return_index = index
        
        # If there are no nn.Module elements, return a random index
        if return_index == -1:
            return np.random.randint(0, len(all_client_latest_deltas))
        return return_index


class RoundRobinAsyncUserSelector(AsyncUserSelector):
    r"""
    Chooses users in round-robin order, starting from user=0.
    Particularly useful for testing.
    """

    def __init__(self, data_provider: IFLDataProvider):
        super().__init__(data_provider)
        self.current_user_index: int = 0

    def get_random_user(self) -> AsyncUserSelectorInfo:
        user_index = self.current_user_index
        self.current_user_index = (
            self.current_user_index + 1
        ) % self.data_provider.num_train_users()
        return AsyncUserSelectorInfo(
            user_data=self.data_provider.get_train_user(user_index),
            user_index=user_index,
        )


class AsyncUserSelectorType(Enum):
    RANDOM = auto()
    ROUND_ROBIN = auto()
    CUSTOM = auto()


class AsyncUserSelectorFactory:
    @classmethod
    def create_users_selector(
        cls, type: AsyncUserSelectorType, data_provider: IFLDataProvider
    ):
        if type == AsyncUserSelectorType.RANDOM:
            return RandomAsyncUserSelector(data_provider)
        elif type == AsyncUserSelectorType.ROUND_ROBIN:
            return RoundRobinAsyncUserSelector(data_provider)
        elif type == AsyncUserSelectorType.CUSTOM:
            return CustomAsyncUserSelector(data_provider)
        else:
            raise AssertionError(f"Unknown user selector type: {type}")
