from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Create PDF
pdf_path = r"C:/Users/kesha/OneDrive/Desktop/Projects/ml-from-scratch/Phase_11_NN/Keshav_Chandel_Resume_Recreated_Final.pdf"

doc = BaseDocTemplate(pdf_path, pagesize=A4,
                      leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)

# Define frames: left column (sidebar) and right column (main)
frame_sidebar = Frame(doc.leftMargin, doc.bottomMargin, 180, doc.height, id='sidebar')
frame_main = Frame(doc.leftMargin + 190, doc.bottomMargin, doc.width - 190, doc.height, id='main')

# Apply the frames to a page template
doc.addPageTemplates([PageTemplate(id='TwoCol', frames=[frame_sidebar, frame_main])])

# Styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle('TitleStyle', fontSize=18, textColor=colors.white, leading=22, spaceAfter=6)
subtitle_style = ParagraphStyle('SubtitleStyle', fontSize=12, textColor=colors.white, leading=14)
section_heading = ParagraphStyle('SectionHeading', fontSize=13, leading=16, spaceAfter=8, spaceBefore=12, fontName="Helvetica-Bold")
bullet_style = ParagraphStyle('BulletStyle', fontSize=10.5, leftIndent=12, spaceAfter=4, leading=14)
body_style = ParagraphStyle('BodyStyle', fontSize=10.5, leading=14)

# Sidebar content (left column)
left_col = [
    Paragraph("KESHAV CHANDEL", title_style),
    Paragraph("B.Tech student", subtitle_style),
    Spacer(1, 12),
    Paragraph("📧 keshavchandel05@gmail.com", subtitle_style),
    Paragraph("📞 +91-8626997510", subtitle_style),
    Paragraph("🔗 linkedin.com/in/keshav-chandel-176b97289", subtitle_style),
    Paragraph("💻 github.com/KESHAV-CHANDEL-07", subtitle_style),
    Spacer(1, 18),
    Paragraph("EDUCATION", section_heading),
    Paragraph("<b>Bachelor of Technology</b><br/>Electronics and Communication Engineering<br/><b>NIT Hamirpur</b><br/>2023 – 2027, current", bullet_style),
    Spacer(1, 8),
    Paragraph("<b>PCM</b><br/><b>HPBOSE</b><br/>2022 – 2023", bullet_style),
    Spacer(1, 18),
    Paragraph("SKILLS", section_heading),
    Paragraph("<b>Libraries & Tools:</b> NumPy, Pandas, Matplotlib, OpenCV, YOLO, scikit-learn", bullet_style),
    Paragraph("<b>Domains:</b> Machine Learning (from scratch), IoT, Deep learning, Computer Vision, Embedded Systems", bullet_style),
    Paragraph("<b>Hardware:</b> Raspberry Pi, Arduino, Sensors, Motors", bullet_style),
    Paragraph("<b>Languages:</b> Python, C, HTML, CSS, JavaScript", bullet_style)
]

# Main content (right column)
right_col = [
    Paragraph("CAREER OBJECTIVE", section_heading),
    Paragraph(
        "Curious and hands-on B.Tech student at NIT Hamirpur, passionate about combining machine learning with embedded hardware. "
        "Focused on building intelligent systems from scratch, exploring real-world applications, and understanding the 'why' behind every concept. "
        "Seeking impactful internship opportunities in AI, IoT, robotics, or applied ML domains to contribute meaningfully and grow.",
        body_style),
    Spacer(1, 12),
    Paragraph("WORK EXPERIENCE", section_heading),
    Paragraph("<b>Executive Member</b><br/>Team Vibhav, <b>NIT Hamirpur</b><br/>Jan 2024 - current", bullet_style),
    Paragraph("• Pioneered automation builds for 3 club tech projects, enhancing efficiency by 40% through Arduino and sensor systems.", bullet_style),
    Paragraph("• Mentored 10+ junior members in electronics and embedded systems, boosting their technical skills by 30% over three months.", bullet_style),
    Paragraph("• Organized 4 hands-on workshops with 100+ participants across departments.", bullet_style),
    Paragraph("• Drove innovative initiatives that led Team Vibhav to win 'Best Departmental Club' among 15 competitors during Nimbus 2k25.", bullet_style),
    Spacer(1, 12),
    Paragraph("PROJECTS", section_heading),
    Paragraph("<b>Machine Learning from Scratch</b> | ML Developer<br/>– current", bullet_style),
    Paragraph("• Implemented core algorithms (linear/logistic regression, decision trees, random forests) using Python and NumPy without external ML libraries.", bullet_style),
    Paragraph("• Achieved 91% accuracy on heart disease dataset; performance matched scikit-learn models within a 3% margin.", bullet_style),
    Paragraph("• Applied gradient descent, entropy, and regularization techniques for mathematical transparency and optimized learning.", bullet_style),
    Spacer(1, 6),
    Paragraph("<b>Waste Segregation Bot</b> | Computer Vision + Robotics Engineer<br/>Jan 2025 – Apr 2025", bullet_style),
    Paragraph("• Developed a YOLOv5-based real-time waste detection system to classify biodegradable vs non-biodegradable materials.", bullet_style),
    Paragraph("• Deployed system on Raspberry Pi with webcam input and a 4-DOF robotic arm, enabling automated sorting.", bullet_style),
    Paragraph("• Achieved 85% detection accuracy; reduced manual waste segregation time by 70%.", bullet_style),
    Spacer(1, 6),
    Paragraph("<b>DIY Air Purifier</b> | Hardware Designer<br/>May 2019 – Feb 2020", bullet_style),
    Paragraph("• Designed a cost-effective air purifier using ionizer fans, HEPA and activated carbon filters.", bullet_style),
    Paragraph("• Tested 3 airflow configurations; reduced PM2.5 concentrations by ~40% in a small indoor environment.", bullet_style),
    Paragraph("• Optimized for use in dorm rooms with total cost under ₹1000.", bullet_style)
]

# Combine the flows
elements = left_col + [PageBreak()] + right_col

# Build the document
doc.build(elements)
