# GitHub Projects - How to Set Up a Gantt Chart

## Overview

GitHub has a built-in project management system called **GitHub Projects** that includes a **timeline view** which functions as a Gantt chart for tracking issues with dates.

## Current Status

The `interactive_ddsp` repository currently has **no GitHub Projects** configured.

## How to Create a Project with Timeline/Gantt View

### Step 1: Create a New Project

1. Go to your repository: https://github.com/buscon/interactive_ddsp
2. Click on the **Projects** tab
3. Click **New project**
4. Choose **Board** or **Timeline** view:
   - **Board** - Kanban-style (To Do, In Progress, Done)
   - **Timeline** - Gantt-style with dates (recommended for this project)

### Step 2: Configure the Timeline View

Once you create a Timeline project, you can:

1. **Add issues** from the repository to the project
2. **Set start/end dates** on each issue:
   - Click on an issue in the project
   - Add start date and due date
   - The timeline will automatically position it

3. **Create milestones** as date markers:
   - M0: Decision and Setup
   - M1: Data Pipeline
   - M2: Model
   - M3: Evaluation
   - M4: Real-time and Delivery

### Step 3: Visual Timeline

The timeline view will show:
- Horizontal bar for each issue spanning its duration
- Milestones as vertical markers
- Dependencies (if configured)
- Progress indicators

## Recommended Project Structure

Based on the current plan, create these milestones in the project:

| Milestone | Target Date | Issues |
|-----------|-------------|--------|
| M0: Decision and Setup | Dec 10, 2026 | #2 |
| M1: Data Pipeline | Dec 17, 2026 | #3, #4, #5 |
| M2: Model | Dec 31, 2026 | #6, #7, #8, #9 |
| M3: Evaluation | Jan 3, 2027 | (depends on M2) |
| M4: Real-time and Delivery | Jan 10, 2027 | #10, #11 |

## Timeline View Features

- **Drag and drop** to reschedule issues
- **Zoom** in/out to see different time ranges
- **Color coding** by priority or phase
- **Status indicators** (progress bars)
- **Date constraints** for dependencies

## Alternative: Use the Project Board + Timeline

You can also create a **Board-style** project and then add the **Timeline** view to it, giving you both:

1. Kanban board for day-to-day work tracking
2. Timeline view for high-level planning and milestone tracking

## Links

- GitHub Projects Documentation: https://docs.github.com/en/issues/organizing-your-work-with-project-boards
- Timeline View Guide: https://docs.github.com/en/issues/planning-and-tracking-with-project-boards/creating-a-timeline-view

## Next Steps

1. Create a GitHub Project for this repository
2. Choose **Timeline** view as the default
3. Add all the GitHub issues to the project
4. Set start/end dates for each issue
5. Create milestones for major phases
6. Use the timeline view for planning and updates

---

**Note:** This is a free feature available to all GitHub users.