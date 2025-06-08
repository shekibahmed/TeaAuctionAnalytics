"""
Tea-themed loading animations and progress indicators for the CTC Tea Sales Analytics Dashboard
"""

import streamlit as st
import time
import random
from typing import List, Optional

class TeaLoadingAnimations:
    """Tea-themed loading animations and progress indicators"""
    
    # Tea-themed messages for different loading stages
    TEA_MESSAGES = {
        'upload': [
            "Brewing your data upload...",
            "Steeping Excel file into memory...",
            "Preparing fresh data for analysis...",
            "Warming up the data processor...",
            "Blending your spreadsheet ingredients..."
        ],
        'processing': [
            "Crushing, tearing, and curling your data...",
            "Sorting tea leaves by market segments...",
            "Analyzing auction house records...",
            "Calculating market efficiency ratios...",
            "Brewing statistical insights...",
            "Filtering data through quality checks...",
            "Optimizing price trend calculations..."
        ],
        'ai_analysis': [
            "AI sommelier tasting your data...",
            "Generating market intelligence brew...",
            "Crafting personalized market insights...",
            "Analyzing flavor profiles of your data...",
            "Distilling complex patterns into wisdom...",
            "Blending market trends with predictions..."
        ],
        'visualization': [
            "Arranging tea leaves into beautiful charts...",
            "Crafting visual tea ceremony...",
            "Painting market landscapes with data...",
            "Designing interactive dashboard elements...",
            "Orchestrating data into visual harmony..."
        ],
        'report': [
            "Packaging insights into PDF blend...",
            "Sealing market analysis in digital containers...",
            "Preparing comprehensive tea market report...",
            "Finalizing statistical brewing summary..."
        ]
    }
    
    # Tea-related emoji and symbols
    TEA_ICONS = ['🍃', '🫖', '☕', '🌱', '💨', '🔥', '⚡', '✨', '💫', '🌟']
    
    @staticmethod
    def get_random_message(category: str) -> str:
        """Get a random tea-themed message for the given category"""
        messages = TeaLoadingAnimations.TEA_MESSAGES.get(category, ['Processing...'])
        return random.choice(messages)
    
    @staticmethod
    def get_random_icon() -> str:
        """Get a random tea-themed icon"""
        return random.choice(TeaLoadingAnimations.TEA_ICONS)
    
    @staticmethod
    def spinning_teapot_animation(duration: float = 2.0, message: str = "Brewing...") -> None:
        """Display a spinning teapot animation"""
        teapot_frames = ['🫖', '🫖', '☕', '☕', '💨', '💨']
        placeholder = st.empty()
        
        start_time = time.time()
        frame_idx = 0
        
        while time.time() - start_time < duration:
            current_frame = teapot_frames[frame_idx % len(teapot_frames)]
            placeholder.markdown(f"<div style='text-align: center; font-size: 2em;'>{current_frame}<br><small>{message}</small></div>", 
                               unsafe_allow_html=True)
            time.sleep(0.3)
            frame_idx += 1
        
        placeholder.empty()
    
    @staticmethod
    def tea_leaf_progress_bar(progress: float, message: str = "Processing...") -> None:
        """Display a tea leaf themed progress bar"""
        # Create progress bar with tea leaf filling
        filled_leaves = int(progress * 20)  # 20 leaves for full progress
        empty_spaces = 20 - filled_leaves
        
        progress_visual = "🍃" * filled_leaves + "⬜" * empty_spaces
        percentage = int(progress * 100)
        
        st.markdown(f"""
        <div style='text-align: center; margin: 20px 0;'>
            <div style='font-size: 1.2em; margin-bottom: 10px;'>{message}</div>
            <div style='font-size: 1.5em; letter-spacing: 2px; margin: 10px 0;'>{progress_visual}</div>
            <div style='font-size: 1.1em; color: #2E8B57;'>{percentage}% Complete</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def steaming_cup_animation(steps: List[str], current_step: int = 0) -> None:
        """Display steaming cup animation with step progress"""
        # Different steam patterns for animation
        steam_patterns = ['💨', '💨💨', '💨💨💨', '💨💨', '💨']
        steam = steam_patterns[current_step % len(steam_patterns)]
        
        # Progress indicator
        progress = (current_step + 1) / len(steps) if steps else 0
        
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: linear-gradient(45deg, #f0f8ff, #e6f3ff); border-radius: 15px; margin: 10px 0;'>
            <div style='font-size: 3em; margin-bottom: 10px;'>☕ {steam}</div>
            <div style='font-size: 1.2em; color: #2F4F4F; margin-bottom: 15px;'>
                Step {current_step + 1} of {len(steps)}
            </div>
            <div style='font-size: 1em; color: #4682B4; margin-bottom: 10px;'>
                {steps[current_step] if current_step < len(steps) else "Complete!"}
            </div>
            <div style='background: #ddd; border-radius: 10px; height: 10px; margin: 15px auto; width: 200px;'>
                <div style='background: linear-gradient(90deg, #32CD32, #228B22); height: 100%; border-radius: 10px; width: {progress*100}%; transition: width 0.3s ease;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def tea_garden_loader(message: str = "Growing insights from your data...") -> None:
        """Display a tea garden growing animation"""
        growth_stages = ['🌱', '🌿', '🍃', '🌳', '🍃✨']
        
        placeholder = st.empty()
        
        for stage in growth_stages:
            placeholder.markdown(f"""
            <div style='text-align: center; padding: 30px;'>
                <div style='font-size: 4em; margin-bottom: 15px;'>{stage}</div>
                <div style='font-size: 1.2em; color: #228B22;'>{message}</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.8)
        
        placeholder.empty()
    
    @staticmethod
    def brewing_process_animation(stages: List[str]) -> None:
        """Display a complete tea brewing process animation"""
        brewing_icons = ['🫖', '💨', '☕', '🍃', '✨']
        
        placeholder = st.empty()
        
        for i, (stage, icon) in enumerate(zip(stages, brewing_icons)):
            progress = (i + 1) / len(stages)
            
            placeholder.markdown(f"""
            <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, #fff8e1, #f3e5ab); border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                <div style='font-size: 3.5em; margin-bottom: 15px;'>{icon}</div>
                <div style='font-size: 1.3em; color: #8B4513; font-weight: bold; margin-bottom: 10px;'>
                    {stage}
                </div>
                <div style='background: #ddd; border-radius: 15px; height: 12px; margin: 20px auto; width: 250px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='background: linear-gradient(90deg, #DAA520, #FF8C00); height: 100%; border-radius: 15px; width: {progress*100}%; transition: width 0.5s ease; box-shadow: 0 2px 8px rgba(218,165,32,0.3);'></div>
                </div>
                <div style='font-size: 1em; color: #A0522D; margin-top: 10px;'>
                    Step {i+1} of {len(stages)}
                </div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.5)
        
        placeholder.empty()

def show_loading_animation(animation_type: str, **kwargs) -> None:
    """Show a specific loading animation"""
    animations = TeaLoadingAnimations()
    
    if animation_type == "spinning_teapot":
        animations.spinning_teapot_animation(**kwargs)
    elif animation_type == "tea_garden":
        animations.tea_garden_loader(**kwargs)
    elif animation_type == "brewing_process":
        animations.brewing_process_animation(**kwargs)
    elif animation_type == "steaming_cup":
        animations.steaming_cup_animation(**kwargs)

class ProgressTracker:
    """Context manager for tracking progress with tea-themed animations"""
    
    def __init__(self, total_steps: int, category: str = 'processing'):
        self.total_steps = total_steps
        self.current_step = 0
        self.category = category
        self.placeholder = None
        self.start_time = time.time()
    
    def __enter__(self):
        self.placeholder = st.empty()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.placeholder:
            self.placeholder.empty()
    
    def update(self, step_name: str = None, increment: bool = True):
        """Update progress with current step"""
        if increment:
            self.current_step += 1
        
        progress = self.current_step / self.total_steps
        
        if step_name is None:
            step_name = TeaLoadingAnimations.get_random_message(self.category)
        
        icon = TeaLoadingAnimations.get_random_icon()
        elapsed_time = time.time() - self.start_time
        
        if self.placeholder:
            self.placeholder.markdown(f"""
            <div style='text-align: center; padding: 20px; background: linear-gradient(45deg, #f8f9fa, #e9ecef); border-radius: 15px; border: 2px solid #28a745; margin: 15px 0;'>
                <div style='font-size: 2.5em; margin-bottom: 10px;'>{icon}</div>
                <div style='font-size: 1.2em; color: #495057; margin-bottom: 15px; font-weight: 500;'>
                    {step_name}
                </div>
                <div style='background: #e9ecef; border-radius: 20px; height: 20px; margin: 15px auto; width: 300px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='background: linear-gradient(90deg, #28a745, #20c997); height: 100%; border-radius: 20px; width: {progress*100}%; transition: width 0.3s ease; box-shadow: 0 2px 10px rgba(40,167,69,0.3);'></div>
                </div>
                <div style='display: flex; justify-content: space-between; margin-top: 10px; font-size: 0.9em; color: #6c757d;'>
                    <span>Step {self.current_step}/{self.total_steps}</span>
                    <span>{int(progress*100)}% Complete</span>
                    <span>{elapsed_time:.1f}s elapsed</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

def simulate_processing_delay(min_delay: float = 0.5, max_delay: float = 1.5) -> None:
    """Simulate realistic processing delay for better UX"""
    delay = random.uniform(min_delay, max_delay)
    time.sleep(delay)