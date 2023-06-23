#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General test class for Injector Class to check
- incoming bunch intensity to the LHC 
"""
from ..injector_model.injector_model import InjectorChain

class testClass_injectorModel:
    
    def test_intensity_into_LHC_vs_Bruce(self):
        
        assert 