import math
from datetime import datetime

from django.db import connection, transaction
from django.db.models import F, CharField, Value, Q
from django.db.models.functions import Concat
from django.shortcuts import render, redirect
from django.utils import timezone
from django.views import View
from rest_framework.response import Response
from rest_framework.views import APIView

from ai.models import AiPostReply
from alarm.models import Alarm
from apply.models import Apply, Trainee
from knowhow.models import Knowhow, KnowhowTag, KnowhowFile, KnowhowRecommend, KnowhowReply, KnowhowCategory, \
    KnowhowPlant, KnowhowScrap, KnowhowLike, KnowhowReplyLike
from lecture.models import Lecture, LectureReview
from member.models import Member
from notice.models import Notice
from order.models import Order, OrderDetail, OrderMileage
from post.models import Post, PostTag, PostFile, PostReply, PostCategory, PostPlant, PostScrap, PostLike, PostReplyLike
from qna.models import QnA
from report.models import KnowhowReplyReport, PostReplyReport, LectureReport, TradeReport, PostReport, KnowhowReport
from selleaf.utils.util import profanityDetectionModel
from teacher.models import Teacher
from trade.models import Trade, TradeCategory


# 메인 헤더 뷰
class HeaderView(View):
    def get(self, request):
        # 알람 - 세션에 있는 member의 id로 회원을 특정하고, 해당 회원이 아직 확인하지 않은 알람 목록을 가져옴
        alarms = Alarm.objects.filter(receiver_id=request.session.get('member').get('id'), alarm_status=False)

        # 위에서 가져온 알람 쿼리셋의 길이(= 미확인 알람 수)를 alarm_count에 저장
        alarm_count = len(alarms)

        # 알람 수를 context 변수에 dict 형태로 저장
        context = {
            'alarm_count': alarm_count,
        }

        # fix/header.html이 열릴 때, alarm_count를 해당 화면에서 사용할 수 있게 함
        return render(request, 'fix/header.html', context)


# 관리자 로그인
class ManagerLoginView(View):
    # 관리자 로그인 페이지 이동 뷰
    def get(self, request):
        # 관리자 로그인이 되어있는 상태에서 다시 관리자 로그인 페이지에 접근하려 했는지 검사
        if request.session.get('admin') is not None:
            # 만약 그렇다면 회원 관리 페이지로 redirect
            return redirect('manager-member')

        # 관리자 로그인이 안 되어있던 상태라면 관리자 로그인 페이지로 이동
        return render(request, 'manager/login/login.html')

    # 관리자 로그인 버튼을 누른 후의 뷰
    def post(self, request):
        # 로그인 정보를 가져옴
        data = request.POST

        # 관리자 로그인 정보를 'admin' 이라는 키로 세션에 저장
        request.session['admin'] = data

        # 이전에 요청한 관리자 페이지 내 경로가 있다면 변수에 담음
        previous_uri = request.session.get('previous_uri')

        # 따로 요청한 경로가 없을 때에는 회원 관리 페이지로 이동
        path = 'manager-member'

        # 만약 따로 요청한 페이지가 있었다면
        if previous_uri is not None:
            # 이동하려고 하는 경로를 기존에 요청한 페이지로 지정
            path = previous_uri

            # 원래 요청했던 페이지에 대한 정보는 세션에서 제거
            del request.session['previous_uri']

        # 위 분기에 따라 결정된 페이지로 이동
        # 기본적으로는 회원 관리 페이지로 이동
        return redirect(path)


# 관리자 로그아웃
class ManagerLogoutView(View):
    def get(self, request):
        # 세션에서 관리자 정보와 기존에 요청했던 페이지에 대한 정보 삭제
        del request.session['admin']

        # 관리자 로그인 페이지로 이동
        return redirect('manager-login')


# 회원 관리
class MemberManagementView(View):
    # 회원 관리 페이지 이동 뷰
    def get(self, request):
        # 현재 가입한 회원 수(휴면 + 비휴면)
        member_count = Member.objects.count()

        # 화면에 보내기 전에 dict 데이터로 만들어 줌
        context = {'member_count': member_count}

        # member.html을 불러오면서 화면에 회원 수 전달
        return render(request, 'manager/member/member/member.html', context)


class MemberInfoAPI(APIView):
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 회원 수
        row_count = 10

        # 한 페이지에 표시할 회원 목록을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 회원 이름 or 이메일 or 주소 or 가입 유형 or 회원 상태 or 가입 일자를 검색
        if keyword:
            condition |= Q(member_name__icontains=keyword)
            condition |= Q(member_email__icontains=keyword)
            condition |= Q(member_address__icontains=keyword)
            condition |= Q(member_type__icontains=keyword)
            condition |= Q(member_status__icontains=keyword)
            condition |= Q(created_date__icontains=keyword)

        # 회원 정보 표시에 필요한 컬럼들
        columns = [
            'id',
            'member_name',  # 회원 이름(닉네임)
            'member_email',  # 회원 이메일
            'member_address',  # 회원 주소(tbl_member_address의 city + district)
            'member_type',  # 회원의 가입 유형(구글, 네이버, 카카오)
            'member_status',  # 회원 상태(휴면/비휴면)
            'created_date'  # 가입 일자
        ]

        # 회원들의 정보를 최근에 가입한 순서대로 가져옴
        # 기존에 tbl_member에 없던 주소는 tbl_member_address에서 가져와서 문자열 합치기
        members = Member.objects \
            .annotate(member_address=Concat(F('memberaddress__address_city'),
                                            Value(" "),
                                            F('memberaddress__address_district'),
                                            output_field=CharField()
                                            )
                      ) \
            .values(*columns).filter(condition, id__isnull=False)

        # 각 회원의 가입 일자를 "YYYY.MM.DD" 형식의 문자열로 변환
        for member in members:
            member['created_date'] = member['created_date'].strftime('%Y.%m.%d')

            # 총 마일리지를 담을 변수(초기값 0)
            order_mileage = 0

            # 특정 회원의 마일리지 내역을 전부 가져옴
            mileage_histories = OrderMileage.objects.filter(member_id=member.get('id'))

            # 마일리지 내역 전체를 순회하면서
            for mileage in mileage_histories:
                # mileage_status가 1이면 해당 마일리지는 order_mileage에 더하고
                if mileage.mileage_status:
                    order_mileage += int(mileage.mileage)
                # mileage_status가 0이면 해당 마일리지는 order_mileage에서 차감함
                else:
                    order_mileage -= int(mileage.mileage)

            # 각각의 회원 정보(dict)에 위의 for문으로 구한 마일리지를 담을 새로운 키(member_mileage) 생성
            member['member_mileage'] = order_mileage

        # 총 회원 수
        total = members.count()

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }

        # 회원 목록을 QuerySet -> list 타입으로 변경하고, 회원 10명씩 슬라이싱(페이지 하나)
        members = list(members[offset:limit])

        # 회원 목록의 맨 뒤에 페이지네이션 정보 추가
        members.append(page_info)

        # 요청한 회원 정보 및 페이지네이션에 사용할 정보 반환
        return Response(members)


class DeleteManyMembersAPI(APIView):
    # 한 번에 여러 명의 회원을 휴면 상태로 변경하는 API 뷰
    def patch(self, request, member_ids):
        # 요청 경로에 담긴 member_ids를 콤마(,)를 기준으로 분리해서 list로 만듬
        member_ids = member_ids.split(',')

        # 위 list의 각 요소를 순회
        for member_id in member_ids:
            # 요소가 빈 문자열이 아닐 때만 tbl_member에서 해당 id를 가진 객체를 가져옴
            if member_id != '':
                member = Member.objects.get(id=member_id)

                # 해당 객체의 status를 1(휴면)으로 만들고
                member.member_status = 1

                # updated_date도 현재 시간으로 갱신
                member.updated_date = timezone.now()

                # 변경 내역 저장
                member.save(update_fields=["member_status", "updated_date"])

        # 요청에 대한 응답 반환
        return Response('success')


# 강사 관리
class TeacherManagementView(View):
    # 강사 관리 페이지 이동 뷰
    def get(self, request):
        # 현재 강사 수
        teachers = Teacher.enabled_objects.count()

        # 현재 강사 신청자 수
        teacher_entries = Teacher.objects.filter(teacher_status=0).count()

        # 강사 수를 화면에서 쓸 수 있게 dict 형태로 만들어줌
        context = {
            "teachers": teachers,
            "teacher_entries": teacher_entries
        }

        # 위의 dict 데이터를 teacher.html에 실어서 보냄
        return render(request, 'manager/teacher/teacher.html', context)


class TeacherEntryManagementView(View):
    # 강사 신청자 관리 페이지 이동 뷰
    def get(self, request):
        # 현재 강사 수
        teachers = Teacher.enabled_objects.count()

        # 현재 강사 신청자 수
        teacher_entries = Teacher.objects.filter(teacher_status=0).count()

        # 강사 수를 화면에서 쓸 수 있게 dict 형태로 만들어줌
        context = {
            "teachers": teachers,
            "teacher_entries": teacher_entries
        }

        # 위의 dict 데이터를 teacher-entries.html에 실어서 보냄
        return render(request, 'manager/teacher/teacher-entries.html', context)


class TeacherInfoAPI(APIView):
    # 강사 정보를 가져오는 API 뷰
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 강사 수
        row_count = 10

        # 한 페이지에 표시할 강사 목록을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 강사 이름 or 주요 강의 or 승인 일자를 검색
        if keyword:
            condition |= Q(teacher_name__icontains=keyword)
            condition |= Q(teacher_info__icontains=keyword)
            condition |= Q(updated_date__icontains=keyword)

        # 강사 정보 표시에 필요한 컬럼들
        columns = [
            'id',
            'teacher_name',  # 강사 이름(= 회원 이름)
            'teacher_info',  # 주요 강의
            'lecture_plan',  # 강의 계획
            'updated_date',  # 승인 일자
        ]

        # 강사들의 정보를 최근에 승인된 순서대로 가져옴
        teachers = Teacher.enabled_objects.annotate(teacher_name=F('member__member_name')) \
            .values(*columns).filter(condition, id__isnull=False).order_by('-updated_date')

        # 각 강사의 승인 일자를 "YYYY.MM.DD" 형식의 문자열로 변환
        for teacher in teachers:
            teacher['updated_date'] = teacher['updated_date'].strftime('%Y.%m.%d')

        # 총 강사 수
        total = teachers.count()

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }

        # 강사 목록을 QuerySet -> list 타입으로 변경하고, 강사 10명씩 슬라이싱(페이지 하나)
        teachers = list(teachers[offset:limit])

        # 강사 목록의 맨 뒤에 페이지네이션 정보 추가
        teachers.append(page_info)

        # 요청한 강사 정보 및 페이지네이션에 사용할 정보 반환
        return Response(teachers)


class TeacherEntriesInfoAPI(APIView):
    # 강사 신청자 정보를 가져오는 API 뷰
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 강사 신청자 수
        row_count = 10

        # 한 페이지에 표시할 강사 신청자 목록을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 강사 신청자 이름 or 주요 강의 or 신청 일자를 검색
        if keyword:
            condition |= Q(teacher_name__icontains=keyword)
            condition |= Q(teacher_info__icontains=keyword)
            condition |= Q(created_date__icontains=keyword)

        # 강사 신청자 정보 표시에 필요한 컬럼들
        columns = [
            'id',
            'teacher_name',  # 강사 신청자 이름(= 회원 이름)
            'teacher_info',  # 주요 강의
            'lecture_plan',  # 강의 계획
            'created_date',  # 신청 일자
        ]

        # 강사 신청자들의 정보를 최근에 신청한 순서대로 가져옴
        teacher_entries = Teacher.objects.filter(teacher_status=0).annotate(teacher_name=F('member__member_name')) \
            .values(*columns).filter(condition, id__isnull=False).order_by('-id')

        # 각 강사 신청자들의 신청 일자를 "YYYY.MM.DD" 형식의 문자열로 변환
        for teacher_entry in teacher_entries:
            teacher_entry['created_date'] = teacher_entry['created_date'].strftime('%Y.%m.%d')

        # 총 강사 신청자 수
        total = teacher_entries.count()

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }

        # 강사 신청자 목록을 QuerySet -> list 타입으로 변경하고, 강사 신청자 10명씩 슬라이싱(페이지 하나)
        teacher_entries = list(teacher_entries[offset:limit])

        # 강사 신청자 목록의 맨 뒤에 페이지네이션 정보 추가
        teacher_entries.append(page_info)

        # 요청한 강사 신청자 정보 및 페이지네이션에 사용할 정보 반환
        return Response(teacher_entries)


class TeacherApprovalAPI(APIView):
    # 강사 신청자 여러 명을 한 번에 승인하는 API 뷰
    def patch(self, request, teacher_ids):
        # 요청 경로에 담긴 teacher_ids를 콤마(,)를 기준으로 분리해서 list로 만듬
        teacher_ids = teacher_ids.split(',')

        # 위 list의 각 요소를 순회
        for teacher_id in teacher_ids:
            # 요소가 빈 문자열이 아닐 때만 tbl_teacher에서 해당 id를 가진 객체를 가져옴
            if teacher_id != '':
                teacher = Teacher.objects.get(id=teacher_id)

                # 해당 객체의 status를 1(승인)으로 만들고
                teacher.teacher_status = 1

                # updated_date도 현재 시간으로 갱신
                teacher.updated_date = timezone.now()

                # 변경 내역 저장
                teacher.save(update_fields=["teacher_status", "updated_date"])

        # 요청에 대한 응답 반환
        return Response('success')


class TeacherDeleteAPI(APIView):
    # 강사 여러 명을 한 번에 차단하는 API 뷰
    def patch(self, request, teacher_ids):
        # 요청 경로에 담긴 teacher_ids를 콤마(,)를 기준으로 분리해서 list로 만듬
        teacher_ids = teacher_ids.split(',')

        # 위 list의 각 요소를 순회
        for teacher_id in teacher_ids:
            # 요소가 빈 문자열이 아닐 때만 tbl_teacher에서 해당 id를 가진 객체를 가져옴
            if teacher_id != '':
                teacher = Teacher.objects.get(id=teacher_id)

                # 해당 객체의 status를 -1(차단)으로 만들고
                teacher.teacher_status = -1

                # updated_date도 현재 시간으로 갱신
                teacher.updated_date = timezone.now()

                # 변경 내역 저장
                teacher.save(update_fields=["teacher_status", "updated_date"])

        # 요청에 대한 응답 반환
        return Response('success')


# 게시물 관리
class PostManagementView(View):
    # 게시물 관리 페이지 이동 뷰
    def get(self, request):
        # 일반, 노하우, 거래 각각의 게시물 수
        post_count = Post.objects.count()
        knowhow_count = Knowhow.objects.count()
        trade_count = Trade.enabled_objects.count()

        # 각 유형 별 게시물 수를 context 변수에 dict 형식으로 담음
        context = {
            'post_count': post_count,
            'knowhow_count': knowhow_count,
            'trade_count': trade_count,
        }

        # post.html 페이지로 이동하면서 context 안에 담긴 게시물 수도 같이 보냄
        return render(request, 'manager/post/post.html', context)


class PostsListAPI(APIView):
    # 일반 게시물 조회 API 뷰
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 일반 게시물 수
        row_count = 10

        # 한 페이지에 표시할 일반 게시물 목록을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 게시물 제목 or 게시물 내용 or 작성자 or 카테고리 명을 검색
        if keyword:
            condition |= Q(post_title__icontains=keyword)
            condition |= Q(post_content__icontains=keyword)
            condition |= Q(member_name__icontains=keyword)
            condition |= Q(category_name__icontains=keyword)

        # 일반 게시물 정보 표시에 필요한 컬럼들
        columns = [
            'id',
            'post_title',  # 일반 게시물 제목
            'post_content',  # 일반 게시물 내용
            'member_name',  # 작성자
            'category_name',  # 일반 게시물 카테고리 명
            'created_date',  # 작성 일자
        ]

        # 일반 게시물 목록을 최신순으로 가져옴
        posts = Post.objects.filter() \
            .annotate(member_name=F('member__member_name'),
                      category_name=F('postcategory__category_name')
                      ) \
            .values(*columns).filter(condition, id__isnull=False)

        # 각 일반 게시물의 작성 일자를 "YYYY.MM.DD" 형식의 문자열로 변환
        for post in posts:
            post['created_date'] = post['created_date'].strftime('%Y.%m.%d')

            # 카테고리 명에 내용이 없으면 '없음' 표시
            if post['category_name'] is None:
                post['category_name'] = '없음'

        # 총 일반 게시물 수
        total = posts.count()

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }

        # 일반 게시물 목록을 QuerySet -> list 타입으로 변경하고, 게시물 10개씩 슬라이싱(페이지 하나)
        posts = list(posts[offset:limit])

        # 일반 게시물 목록의 맨 뒤에 페이지네이션 정보 추가
        posts.append(page_info)

        # 요청한 일반 게시물 정보 및 페이지네이션에 사용할 정보 반환
        return Response(posts)


class KnowhowPostsAPI(APIView):
    # 노하우 게시물 조회 API 뷰
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 노하우 게시물 수
        row_count = 10

        # 한 페이지에 표시할 노하우 게시물 목록을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 게시물 제목 or 게시물 내용 or 작성자 or 카테고리 명을 검색
        if keyword:
            condition |= Q(knowhow_title__icontains=keyword)
            condition |= Q(knowhow_content__icontains=keyword)
            condition |= Q(member_name__icontains=keyword)
            condition |= Q(category_name__icontains=keyword)

        # 노하우 게시물 정보 표시에 필요한 컬럼들
        columns = [
            'id',
            'knowhow_title',  # 노하우 게시물 제목
            'knowhow_content',  # 노하우 게시물 내용
            'member_name',  # 작성자
            'category_name',  # 노하우 게사물 카테고리 명
            'created_date',  # 작성 일자
        ]

        # 노하우 게시물 목록을 최신순으로 가져옴
        knowhow_posts = Knowhow.objects.filter() \
            .annotate(member_name=F('member__member_name'),
                      category_name=F('knowhowcategory__category_name')
                      ) \
            .values(*columns).filter(condition, id__isnull=False)

        # 각 노하우 게시물의 작성 일자를 "YYYY.MM.DD" 형식의 문자열로 변환
        for knowhow_post in knowhow_posts:
            knowhow_post['created_date'] = knowhow_post['created_date'].strftime('%Y.%m.%d')

            # 카테고리 명에 내용이 없으면 '없음' 표시
            if knowhow_post['category_name'] is None:
                knowhow_post['category_name'] = '없음'

        # 총 노하우 게시물 수
        total = knowhow_posts.count()

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }

        # 노하우 게시물 목록을 QuerySet -> list 타입으로 변경하고, 게시물 10개씩 슬라이싱(페이지 하나)
        knowhow_posts = list(knowhow_posts[offset:limit])

        # 노하우 게시물 목록의 맨 뒤에 페이지네이션 정보 추가
        knowhow_posts.append(page_info)

        # 요청한 노하우 게시물 정보 및 페이지네이션에 사용할 정보 반환
        return Response(knowhow_posts)


class TradePostsAPI(APIView):
    # 거래 게시물 조회 API 뷰
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 거래 게시물 수
        row_count = 10

        # 한 페이지에 표시할 거래 게시물 목록을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 게시물 제목 or 게시물 내용 or 작성자 or 카테고리 명을 검색
        if keyword:
            condition |= Q(trade_title__icontains=keyword)
            condition |= Q(trade_content__icontains=keyword)
            condition |= Q(member_name__icontains=keyword)
            condition |= Q(category_name__icontains=keyword)

        # 거래 게시물 정보 표시에 필요한 컬럼들
        columns = [
            'id',
            'trade_title',  # 거래 게시물 제목
            'trade_content',  # 거래 게시물 내용
            'member_name',  # 작성자
            'category_name',  # 거래 게시물 카테고리 명
            'created_date',  # 작성 일자
        ]

        # 거래 게시물 목록을 최신순으로 가져옴
        trade_posts = Trade.enabled_objects.filter() \
            .annotate(member_name=F('member__member_name'),
                      category_name=F('trade_category__category_name')) \
            .values(*columns).filter(condition, id__isnull=False)

        # 각 거래 게시물의 작성 일자를 "YYYY.MM.DD" 형식의 문자열로 변환
        for trade_post in trade_posts:
            trade_post['created_date'] = trade_post['created_date'].strftime('%Y.%m.%d')

        # 총 거래 게시물 수
        total = trade_posts.count()

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }

        # 거래 게시물 목록을 QuerySet -> list 타입으로 변경하고, 게시물 10개씩 슬라이싱(페이지 하나)
        trade_posts = list(trade_posts[offset:limit])

        # 거래 게시물 목록의 맨 뒤에 페이지네이션 정보 추가
        trade_posts.append(page_info)

        # 요청한 거래 게시물 정보 및 페이지네이션에 사용할 정보 반환
        return Response(trade_posts)


class PostsDeleteAPI(APIView):
    # 일반 게시물 여러 개를 한 번에 삭제하는 API 뷰
    # transaction.atomic: 메소드 실행 중간에 오류 발생 시, 실행 전 상태로 롤백
    @transaction.atomic
    def delete(self, request, post_ids):
        # 요청 경로에 담긴 post_ids를 콤마(,)를 기준으로 분리해서 list로 만듬
        post_ids = post_ids.split(',')

        # 위 list의 각 요소를 순회
        for post_id in post_ids:
            # 요소가 빈 문자열이 아닐 때만 tbl_post에서 해당 id를 가진 객체를 가져옴
            if post_id != '':
                # 해당 일반 게시물 및 해당 게시물과 연결된 다른 테이블의 정보까지 전부 delete
                PostTag.objects.filter(post_id=post_id).delete()
                PostFile.objects.filter(post_id=post_id).delete()
                PostReplyLike.objects.filter(post_reply__post_id=post_id).delete()
                PostReplyReport.object.filter(post_reply__post_id=post_id).delete()
                PostReply.objects.filter(post_id=post_id).delete()
                PostCategory.objects.filter(post_id=post_id).delete()
                PostPlant.objects.filter(post_id=post_id).delete()
                PostScrap.objects.filter(post_id=post_id).delete()
                PostLike.objects.filter(post_id=post_id).delete()
                Post.objects.filter(id=post_id).delete()

        # 요청에 대한 응답 반환
        return Response('success')


class KnowhowDeleteAPI(APIView):
    # 노하우 게시물 여러 개를 한 번에 삭제하는 API 뷰
    @transaction.atomic
    def delete(self, request, knowhow_ids):
        # 요청 경로에 담긴 knowhow_ids를 콤마(,)를 기준으로 분리해서 list로 만듬
        knowhow_ids = knowhow_ids.split(',')

        # 위 list의 각 요소를 순회
        for knowhow_id in knowhow_ids:
            # 요소가 빈 문자열이 아닐 때만 tbl_knowhow에서 해당 id를 가진 객체를 가져옴
            if knowhow_id != '':
                # 해당 노하우 게시물 및 해당 게시물과 연결된 다른 테이블의 정보까지 전부 delete
                KnowhowTag.objects.filter(knowhow_id=knowhow_id).delete()
                KnowhowFile.objects.filter(knowhow_id=knowhow_id).delete()
                KnowhowRecommend.objects.filter(knowhow_id=knowhow_id).delete()
                KnowhowReplyLike.objects.filter(knowhow_reply__knowhow_id=knowhow_id).delete()
                KnowhowReplyReport.object.filter(knowhow_reply__knowhow_id=knowhow_id).delete()
                KnowhowReply.objects.filter(knowhow_id=knowhow_id).delete()
                KnowhowCategory.objects.filter(knowhow_id=knowhow_id).delete()
                KnowhowPlant.objects.filter(knowhow_id=knowhow_id).delete()
                KnowhowScrap.objects.filter(knowhow_id=knowhow_id).delete()
                KnowhowLike.objects.filter(knowhow_id=knowhow_id).delete()
                Knowhow.objects.filter(id=knowhow_id).delete()

        # 요청에 대한 응답 반환
        return Response('success')


class TradeDeleteAPI(APIView):
    # 거래 게시물 여러 개를 한 번에 삭제하는 API 뷰 - 소프트 딜리트
    @transaction.atomic
    def patch(self, request, trade_ids):
        # 요청 경로에 담긴 trade_ids를 콤마(,)를 기준으로 분리해서 list로 만듬
        trade_ids = trade_ids.split(',')

        # 위 list의 각 요소를 순회
        for trade_id in trade_ids:
            # 요소가 빈 문자열이 아닐 때만 tbl_trade에서 해당 id를 가진 객체를 가져옴
            if trade_id != '':
                trade = Trade.objects.get(id=trade_id)

                # 해당 객체의 status를 0(삭제)으로 만들고
                trade.status = 0

                # updated_date를 현재 시간으로 갱신
                trade.updated_date = timezone.now()

                # 변경 내역 저장
                trade.save(update_fields=["status", "updated_date"])

        # 요청에 대한 응답 반환
        return Response('success')


class PostsCountAPI(APIView):
    # 커뮤니티 게시물 개수를 세는 API
    def get(self, request):
        post_count = Post.objects.count()

        return Response(post_count)


class KnowhowCountAPI(APIView):
    # 노하우 게시물 개수를 세는 API
    def get(self, request):
        knowhow_count = Knowhow.objects.count()

        return Response(knowhow_count)


class TradeCountAPI(APIView):
    # 거래 게시물 개수를 세는 API
    def get(self, request):
        trade_count = Trade.enabled_objects.count()

        return Response(trade_count)


# 강의 관리
class LectureManagementView(View):
    # 강의 관리 페이지 이동 뷰
    def get(self, request):
        # 강의 게시물 전체 개수
        lecture_count = Lecture.objects.filter(lecture_status=0).count()

        # 강의 게시물 수를 context에 담음
        context = {'lecture_count': lecture_count}

        # lecture.html 페이지로 이동하면서 context 안에 담긴 강의 게시물 수도 같이 보냄
        return render(request, 'manager/lecture/lecture/lecture.html', context)


class LectureInfoAPI(APIView):
    # 개설된 강의 정보를 가져오는 API 뷰
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 강의 게시물 수
        row_count = 10

        # 한 페이지에 표시할 강의 목록을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 강의 제목 or 강의 내용 or 강사 이름 or 강의 장소 or 온라인 여부를 검색
        if keyword:
            condition |= Q(lecture_title__icontains=keyword)
            condition |= Q(lecture_content__icontains=keyword)
            condition |= Q(teacher_name__icontains=keyword)
            condition |= Q(lecture_place__icontains=keyword)
            condition |= Q(online_status__icontains=keyword)

        # 강의 정보 표시에 필요한 컬럼들
        columns = [
            'id',
            'lecture_title',  # 강의 제목
            'lecture_content',  # 강의 내용
            'teacher_name',  # 강사 이름(회원 이름)
            'lecture_headcount',  # 강의 정원
            'lecture_price',  # 수강료
            'lecture_place',  # 강의 장소
            'online_status',  # 온라인 여부
            'created_date',  # 강의 개설 일자
        ]

        # 강의 게시물 정보를 최신순으로 가져옴
        lectures = Lecture.objects \
            .annotate(teacher_name=F('teacher__member__member_name'),
                      lecture_place=Concat(F('lectureaddress__address_city'),
                                           Value(" "),
                                           F('lectureaddress__address_district'),
                                           output_field=CharField()
                                           ),
                      ) \
            .values(*columns).filter(condition, lecture_status=0, id__isnull=False)

        # 각 강의 게시물의 작성 일자를 "YYYY.MM.DD" 형식의 문자열로 변환
        for lecture in lectures:
            lecture['created_date'] = lecture['created_date'].strftime('%Y.%m.%d')

            # 각 강의 별 수강생 수 추가
            # 해당 강의를 신청한 내역을 우선 조회

            # 해당 강의를 신청한 내역인가
            apply_condition_lecture = Q(lecture_id=lecture['id'])

            # 강의 신청 상태가 0(신청 완료), 1(수강 완료), 2(신청 중) 중에 하나인가
            apply_condition_vaild = Q(apply_status=0) | Q(apply_status=1) | Q(apply_status=2)

            # 위 2가지를 전부 만족하는 것을 조건식으로 설정
            apply_condition = apply_condition_lecture & apply_condition_vaild

            # 위 조건식으로 해당 강의를 신청한 내역 전체를 조회
            applies = Apply.objects.filter(apply_condition)

            # 각 강의 별 수강생 수를 담을 변수(초기값 0)
            total_trainees = 0

            # 해당 강의의 신청 내역 전부를 순회
            for apply in applies:
                # 각 신청 내역의 수강생 수를 조회
                trainee_count = Trainee.objects.filter(apply=apply.id).count()

                # 전체 수강생 수에 합산
                total_trainees += trainee_count

            # 각각의 강의 내역(dict)에 위의 for문으로 구한 신청자 수를 담을 새로운 키(total_trainees) 생성
            lecture['total_trainees'] = total_trainees

        # 현재 개설된 강의 수
        total = lectures.count()

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }

        # 강의 목록을 QuerySet -> list 타입으로 변경하고, 강의 10개씩 슬라이싱(페이지 하나)
        lectures = list(lectures[offset:limit])

        # 강의 목록의 맨 뒤에 페이지네이션 정보 추가
        lectures.append(page_info)

        # 요청한 강의 정보 및 페이지네이션에 사용할 정보 반환
        return Response(lectures)


# 강의 리뷰 관리
class LectureReviewManagementView(View):
    # 강의 리뷰 관리 페이지 이동 뷰
    def get(self, request):
        # 특정 강의의 id를 쿼리 스트링에서 가져옴
        lecture = Lecture.objects.get(id=request.GET['id'])

        # 해당 강의에 달린 리뷰 개수
        review_count = lecture.lecturereview_set.count()

        # 각 강의 별 수강생 수 추가
        # 해당 강의를 신청한 내역을 우선 조회

        # 해당 강의를 신청한 내역인가
        apply_condition_lecture = Q(lecture_id=lecture.id)

        # 강의 신청 상태가 0(신청 완료), 1(수강 완료), 2(신청 중) 중에 하나인가
        apply_condition_vaild = Q(apply_status=0) | Q(apply_status=1) | Q(apply_status=2)

        # 위 2가지를 전부 만족하는 것을 조건식으로 설정
        apply_condition = apply_condition_lecture & apply_condition_vaild

        # 위 조건식으로 해당 강의를 신청한 내역 전체를 조회
        applies = Apply.objects.filter(apply_condition)

        # 각 강의 별 수강생 수를 담을 변수(초기값 0)
        total_trainees_count = 0

        # 각각의 신청 내역에서 수강생 수 구하기
        for apply in applies:
            trainees_count = Trainee.objects.filter(apply=apply.id).count()

            # 총 수강생 수에 합산
            total_trainees_count += trainees_count

        # 강의 정보와 리뷰 개수, 수강생 수를 context에 담음
        context = {
            'lecture': lecture,
            'review_count': review_count,
            'trainees_count': total_trainees_count
        }

        # 아래의 html 페이지로 이동하면서, context에 담긴 정보들을 같이 가져감
        return render(request, 'manager/lecture/lecture-detail/lecture-detail-review.html', context)


class LectureReviewInfoAPI(APIView):
    # 특정 강의에 달린 리뷰 목록을 가져오는 API 뷰
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        lecture_id = request.GET.get('lectureId')
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 강의 리뷰의 수
        row_count = 10

        # 한 페이지에 표시할 강의 리뷰 목록을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 리뷰 제목 or 리뷰 내용 or 작성자 or 별점을 검색
        if keyword:
            condition |= Q(review_title__icontains=keyword)
            condition |= Q(review_content__icontains=keyword)
            condition |= Q(member_name__icontains=keyword)
            condition |= Q(review_rating__icontains=keyword)

        # 리뷰 정보 표시에 필요한 컬럼들
        columns = [
            'id',
            'review_title',  # 리뷰 제목
            'review_content',  # 리뷰 내용
            'member_name',  # 작성자
            'review_rating',  # 별점
            'created_date',  # 작성 일자
        ]

        # 특정 강의의 리뷰 목록을 최신순으로 가져옴
        reviews = LectureReview.objects.filter(lecture=lecture_id) \
            .annotate(member_name=F('member__member_name')) \
            .values(*columns).filter(condition, id__isnull=False)

        # 각 강의 리뷰의 작성 일자를 "YYYY.MM.DD" 형식의 문자열로 변환
        for review in reviews:
            review['created_date'] = review['created_date'].strftime('%Y.%m.%d')

        # 특정 강의의 총 리뷰 개수
        total = reviews.count()

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }

        # 리뷰 목록을 QuerySet -> list 타입으로 변경하고, 리뷰 10개씩 슬라이싱(페이지 하나)
        reviews = list(reviews[offset:limit])

        # 리뷰 목록의 맨 뒤에 페이지네이션 정보 추가
        reviews.append(page_info)

        # 요청한 강의 리뷰 정보 및 페이지네이션에 사용할 정보 반환
        return Response(reviews)


# 강의 수강생 리스트 관리
class LectureTraineesManagementView(View):
    # 강의 수강생 관리 페이지 이동 뷰
    def get(self, request):
        # 특정 강의의 id를 쿼리 스트링에서 가져옴
        lecture = Lecture.objects.get(id=request.GET['id'])

        # 해당 강의에 달린 리뷰 개수
        review_count = lecture.lecturereview_set.count()

        # 각 강의 별 수강생 수 추가
        # 해당 강의를 신청한 내역을 우선 조회

        # 해당 강의를 신청한 내역인가
        apply_condition_lecture = Q(lecture_id=lecture.id)

        # 강의 신청 상태가 0(신청 완료), 1(수강 완료), 2(신청 중) 중에 하나인가
        apply_condition_vaild = Q(apply_status=0) | Q(apply_status=1) | Q(apply_status=2)

        # 위 2가지를 전부 만족하는 것을 조건식으로 설정
        apply_condition = apply_condition_lecture & apply_condition_vaild

        # 위 조건식으로 해당 강의를 신청한 내역 전체를 조회
        applies = Apply.objects.filter(apply_condition)

        # 각 강의 별 수강생 수를 담을 변수(초기값 0)
        total_trainees_count = 0

        # 각각의 신청 내역에서 수강생 수 구하기
        for apply in applies:
            trainees_count = Trainee.objects.filter(apply=apply.id).count()

            # 총 수강생 수에 합산
            total_trainees_count += trainees_count

        # 강의 정보와 리뷰 개수, 수강생 수를 context에 담음
        context = {
            'lecture': lecture,
            'review_count': review_count,
            'trainees_count': total_trainees_count
        }

        # 아래의 html 페이지로 이동하면서, context에 담긴 정보들을 같이 가져감
        return render(request, 'manager/lecture/lecture-detail/lecture-detail-student.html', context)


class TraineesInfoAPI(APIView):
    # 특정 강의의 수강생 목록을 조회하는 API 뷰
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        lecture_id = request.GET.get('lectureId')
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 수강생 수
        row_count = 10

        # 한 페이지에 표시할 수강생 목록을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 수강생 이름 or 대표 신청자 or 수강 날짜 or 수강 시간 or 수강 상태 검색
        if keyword:
            condition |= Q(trainee_name__icontains=keyword)
            condition |= Q(main_applicant__icontains=keyword)
            condition |= Q(apply_date__icontains=keyword)
            condition |= Q(apply_time__icontains=keyword)
            condition |= Q(apply_status__icontains=keyword)

        # 수강생 정보 표시에 필요한 컬럼들
        columns = [
            'id',
            'trainee_name',  # 수강생 이름
            'main_applicant',  # 대표 신청자(회원)
            'apply_date',  # 수강 날짜
            'apply_time',  # 수강 시간
            'apply_status',  # 0(신청 완료), 1(수강 완료), -1(수강 취소)
            'purchase_date',  # 수강 신청일
        ]

        # 특정 강의를 신청한 수강 신청 내역을 가져옴
        applies = Apply.objects.filter(lecture=lecture_id)

        # 수강 신청 내역을 받아올 dict를 미리 만들어줌
        trainees = {}

        for apply in applies:
            # 특정 강의의 수강생 목록을 최신순으로 가져옴
            trainees = Trainee.objects.filter(apply=apply.id) \
                .annotate(main_applicant=F('apply__member__member_name'),
                          apply_date=F('apply__date'),
                          apply_time=F('apply__time'),
                          apply_status=F('apply__apply_status'),
                          purchase_date=F('apply__created_date')) \
                .values(*columns).filter(condition, id__isnull=False)

            # 각 수강생의 수강 신청일을 "YYYY.MM.DD" 형식의 문자열로 변환
            for trainee in trainees:
                trainee['purchase_date'] = trainee['purchase_date'].strftime('%Y.%m.%d')

        # 강의 수강생 수
        total = trainees.count()

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }

        # 수강생 목록을 QuerySet -> list 타입으로 변경하고, 수강생 10명씩 슬라이싱(페이지 하나)
        trainees = list(trainees[offset:limit])

        # 수강생 목록의 맨 뒤에 페이지네이션 정보 추가
        trainees.append(page_info)

        # 요청한 강의 수강생 정보 및 페이지네이션에 사용할 정보 반환
        return Response(trainees)


# 강의 여러 개를 한 번에 삭제하는 뷰 - 소프트 딜리트
class LectureDeleteAPI(APIView):
    def patch(self, request, lecture_ids):
        # 요청 경로에 담긴 lecture_ids를 콤마(,)를 기준으로 분리해서 list로 만듬
        lecture_ids = lecture_ids.split(',')

        # 위 list의 각 요소를 순회
        for lecture_id in lecture_ids:
            # 요소가 빈 문자열이 아닐 때만 tbl_lecture에서 해당 id를 가진 객체를 가져옴
            if lecture_id != '':
                notice = Lecture.objects.get(id=lecture_id)

                # 해당 객체의 status를 0(삭제)으로 만들고
                notice.lecture_status = 0

                # updated_date를 현재 시간으로 갱신
                notice.updated_date = timezone.now()

                # 변경 내역 저장
                notice.save(update_fields=["lecture_status", "updated_date"])

        # 요청에 대한 응답 반환
        return Response('success')


# 강의 리뷰 여러 개를 한 번에 삭제하는 API 뷰
class LectureReviewDeleteAPI(APIView):
    def delete(self, request, lecture_review_ids):
        # 요청 경로에 담긴 lecture_review_id를 콤마(,)를 기준으로 분리해서 list로 만듬
        lecture_review_ids = lecture_review_ids.split(',')

        # 위 list의 각 요소를 순회
        for lecture_review_id in lecture_review_ids:
            # 요소가 빈 문자열이 아닐 때만 tbl_lecture_review에서 해당 id를 가진 객체를 가져와서 삭제
            if lecture_review_id != '':
                LectureReview.objects.get(id=lecture_review_id).delete()

        # 요청에 대한 응답 반환
        return Response('success')


# 댓글 관리
class ReplyManagementView(View):
    # 댓글 관리 페이지 이동 뷰
    def get(self, request):
        return render(request, 'manager/comment/comment.html')


# 댓글 목록 조회, 댓글 삭제를 담당하는 API 뷰
class ReplyManagementAPI(APIView):
    # 댓글 목록을 조회하는 메소드
    def get(self, request):
        # 쿼리 스트링에서 키워드와 페이지 정보를 가져옴
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))
        print(keyword)
        # 한 페이지에 표시할 댓글 수
        row_count = 10

        # 댓글 목록을 슬라이싱하기 위해 필요한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 댓글 작성자 이름 or 댓글 내용 검색
        if keyword:
            condition |= Q(reply_member_name__icontains=keyword)
            condition |= Q(reply_content__icontains=keyword)

        # 댓글 정보 표시에 필요한 컬럼들
        columns = [
            'reply_member_id',  # 댓글 작성자 id
            'reply_member_name',  # 댓글 작성자 이름
            'target_title',  # 댓글이 작성된 게시물의 제목
            'reply_id',  # 댓글 id
            'reply_content',  # 댓글 내용
            'reply_created',  # 댓글 작성 일자
        ]

        # 일반 게시물 댓글 목록을 최신순으로 가져옴
        post_replies = Post.objects.annotate(
            reply_member_id=F('postreply__member_id'),
            reply_member_name=F('postreply__member__member_name'),
            target_title=F('post_title'),
            reply_id=F('postreply__id'),
            reply_content=F('postreply__post_reply_content'),
            reply_created=F('postreply__created_date')
        ).values(*columns).filter(condition, reply_member_id__isnull=False)

        # 일반 게시물 댓글 각각에 'target_type'이라는 키를 추가하고, 값으로 '일반 게시물'을 할당
        for post_reply in post_replies:
            post_reply['target_type'] = '일반 게시물'

        # 노하우 댓글 목록을 최신순으로 가져옴
        knowhow_replies = Knowhow.objects.annotate(
            reply_member_id=F('knowhowreply__member_id'),
            reply_member_name=F('knowhowreply__member__member_name'),
            target_title=F('knowhow_title'),
            reply_id=F('knowhowreply__id'),
            reply_content=F('knowhowreply__knowhow_reply_content'),
            reply_created=F('knowhowreply__created_date')
        ).values(*columns).filter(condition, reply_member_id__isnull=False)

        # 노하우 게시물 댓글 각각에 'target_type'이라는 키를 추가하고, 값으로 '노하우'를 할당
        for knowhow_reply in knowhow_replies:
            knowhow_reply['target_type'] = '노하우'

        # 전체 댓글 수
        total = post_replies.union(knowhow_replies).count()

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }

        # 댓글 목록을 합져서 최신순으로 정렬하고 list로 만든 후, 댓글 10개씩 슬라이싱(페이지 하나)
        # 댓글 정보를 불러올 때 annotate를 사용해서 컬럼명을 똑같이 맞춰줬기 때문에 union이 가능
        replies = list(post_replies.union(knowhow_replies).order_by('-reply_created')[offset:limit])

        # 각 댓글 리스트를 순회하면서 '일반 게시물'과 '노하우'로 'target_type 구분'
        for reply in replies:
            for post_reply in post_replies:
                if reply['reply_created'] == post_reply['reply_created']:
                    reply['target_type'] = '일반 게시물'

        for reply in replies:
            for knowhow_reply in knowhow_replies:
                if reply['reply_created'] == knowhow_reply['reply_created']:
                    reply['target_type'] = '노하우'

        # 댓글 목록의 맨 뒤에 페이지네이션에 필요한 정보 추가
        replies.append(page_info)
        print(replies)
        print(1111)
        # 요청한 댓글 정보 및 페이지네이션에 사용할 정보 반환
        return Response(replies)

    # 댓글 여러 개를 한 번에 삭제하는 메소드
    @transaction.atomic
    def delete(self, request):
        # 삭제 요청한 댓글들의 정보를 data 변수에 할당
        datas = request.data

        # 삭제 요청 각각을 순회하면서 댓글 작성자의 id, 댓글 작성 시간, 댓글이 작성된 게시물 유형을 가져옴
        for data in datas:
            reply_member_id = data.get('reply_member_id')
            reply_created = datetime.strptime(data.get('reply_created'), "%Y-%m-%dT%H:%M:%S.%f")
            target_type = data.get('target_type')

            # 게시물 유형이 '일반 게시물'이면 일반 게시물 댓글 테이블에서 해당 댓글 정보 삭제
            if target_type == '일반 게시물':
                PostReply.objects.filter(member_id=reply_member_id, created_date=reply_created).delete()
            # 게시물 유형이 '노하우'면 노하우 게시물 댓글 테이블에서 해당 댓글 정보 삭제
            else:
                KnowhowReply.objects.filter(member_id=reply_member_id, created_date=reply_created).delete()

        # 요청에 대한 응답 반환
        return Response('success')


# 태그 관리
class TagManagementView(View):
    # 태그 관리 페이지 이동 뷰
    def get(self, request):
        return render(request, 'manager/tag/tag.html')


# 태그 목록 조회, 태그 삭제를 담당하는 API 뷰
class TagManagementAPI(APIView):
    # 태그 목록을 조회하는 메소드
    def get(self, request):
        # 쿼리 스트링에서 키워드와 페이지 정보를 가져옴
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 표시할 태그 수
        row_count = 10

        # 태그 목록을 슬라이싱하기 위해 필요한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword에 뭐라도 받았다면, keyword가 포함된 태그명 검색
        if keyword:
            condition |= Q(tag_name__icontains=keyword)

        # 일반 게시물 전체의 태그명을 중복 제거하고 가져옴
        post_tags = PostTag.objects.filter(condition).values('tag_name').distinct()

        # 노하우 게시물 전체의 태그명을 중복 제거하고 가져옴
        knowhow_tags = KnowhowTag.objects.filter(condition).values('tag_name').distinct()

        # 전체 태그 수
        total = post_tags.union(knowhow_tags).count()

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }

        # 태그 목록 전체를 가나다 순으로 정렬하고 list로 만든 후, 태그 10개씩 슬라이싱(페이지 하나)
        tags = list(post_tags.union(knowhow_tags).order_by('tag_name')[offset:limit])

        # 태그 목록의 맨 뒤에 페이지네이션에 필요한 정보 추가
        tags.append(page_info)

        # 요청한 태그 정보 및 페이지네이션에 사용할 정보 반환
        return Response(tags)

    # 태그 여러 개를 한 번에 삭제하는 메소드
    def delete(self, request):
        # 요청에서 석제할 태그들의 정보를 가져와서 data 변수에 할당
        datas = request.data

        # 삭제 요청 각각을 순회하면서 삭제할 태그의 태그명을 가져옴
        for data in datas:
            tag_name = data.get('tag_name')

            # 일반 게시물과 노하우 게시물의 태그 목록에서 해당 이름을 가진 태그를 전부 삭제
            PostTag.objects.filter(tag_name=tag_name).delete()
            KnowhowTag.objects.filter(tag_name=tag_name).delete()

        # 요청에 대한 응답 반환
        return Response('success')


# 결제 내역 관리
class PaymentManagementView(View):
    # 결제 내역 관리 페이지 이동 뷰
    def get(self, request):
        # 현재 발생한 결제 내역의 총 개수
        payment_count = Order.objects.count()

        # 결제 내역 개수를 context에 dict 형식으로 담음
        context = {
            'payment_count': payment_count
        }

        # payment.html로 이동하면서 위의 결재 내역 개수를 가져감
        return render(request, 'manager/payment/payment.html', context)


class PaymentListAPI(APIView):
    # 결제 내역을 가져오는 API 뷰
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 결제 내역 수
        row_count = 10

        # 한 페이지에 표시할 결제 내역들을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 구매자 or 구매 강의 or 키트 or 배송지 or 결제 일자를 검색
        if keyword:
            condition |= Q(payment_member__icontains=keyword)
            condition |= Q(payment_lecture__icontains=keyword)
            condition |= Q(payment_kit__icontains=keyword)
            condition |= Q(payment_address__icontains=keyword)
            condition |= Q(created_date__icontains=keyword)

        # 결제 내역 표시에 필요한 컬럼들
        columns = [
            'id',
            'payment_member',  # 구매자
            'payment_lecture',  # 구매 강의
            'payment_price',  # 수강료
            'payment_kit',  # 구매 키트 - 오프라인은 키트 없음
            'payment_quantity',  # 구매 개수
            'payment_address',  # 배송지
            'payment_status',  # 결제 상태
            'created_date',  # 결제 요청 발생 일자
        ]

        # 결제 내역을 최신순으로 가져옴
        payments = OrderDetail.objects.filter() \
            .annotate(payment_member=F('order__member__member_name'),
                      payment_lecture=F('apply__lecture__lecture_title'), \
                      payment_price=F('apply__lecture__lecture_price'),
                      payment_kit=F('order__kit__kit_name'),
                      payment_quantity=F('apply__quantity'),
                      payment_address=Concat(F('order__address__address_city'),
                                             Value(" "),
                                             F('order__address__address_district'),
                                             output_field=CharField()
                                             ),
                      payment_status=F('order_status')) \
            .values(*columns).filter(condition, id__isnull=False)

        # 각 결제 내역의 요청 발생 일자를 "YYYY.MM.DD" 형식의 문자열로 변환
        for payment in payments:
            payment['created_date'] = payment['created_date'].strftime('%Y.%m.%d')

        # 결제 내역의 총 개수
        total = payments.count()

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
        }

        # 결제 내역 목록을 QuerySet -> list 타입으로 변경하고, 결제 내역 10개씩 슬라이싱(페이지 하나)
        payments = list(payments[offset:limit])

        # 결제 내역 목록의 맨 뒤에 페이지네이션 정보 추가
        payments.append(page_info)

        # 요청한 결제 내역 및 페이지네이션에 사용할 정보 반환
        return Response(payments)


# 공지사항 관리
class NoticeManagementView(View):
    # 공지사항 내역 페이지 이동 뷰
    def get(self, request):
        # 현재 작성된 공지사항 및 QnA의 개수를 세서 dict 데이터로 통합
        notice_count = Notice.enabled_objects.count()
        qna_count = QnA.enabled_objects.count()

        context = {
            'notice_count': notice_count,
            'qna_count': qna_count
        }

        # 공지사항과 QnA 개수를 화면에 전달
        # 공지사항 내역을 가져오는 것은 별도의 API가 해줌
        return render(request, 'manager/manager-notice/manager-notice/manager-notice.html', context)


class WriteNoticeView(View):
    # 공지사항 작성 페이지 이동 뷰
    def get(self, request):
        # 로그인 검사는 미들웨어가 해주기 때문에, 별도의 정보 필요 없이 바로 render
        return render(request, 'manager/manager-notice/manager-notice/manager-notice-compose.html')

    # 공지사항 작성 완료 이후의 뷰
    @transaction.atomic
    def post(self, request):
        # POST 방식으로 요청한 데이터를 가져옴
        notice_data = request.POST

        # 받아온 데이터에서 특정 정보(id, 제목, 내용)를 가져와서 dict 타입으로 저장
        data = {
            'notice_title': notice_data['notice-title'],
            'notice_content': notice_data['notice-content'],
        }

        # 받아온 데이터로 tbl_notice에 실행할 insert 쿼리 작성
        Notice.objects.create(**data)

        # 작성한 공지사항 저장 후, 공지사항 리스트 페이지로 redirect
        return redirect('manager-notice')


class UpdateNoticeView(View):
    # 공지사항 수정 페이지 이동 뷰
    def get(self, request):
        # 수정 버튼에서 전달한 id를 통해 수정할 공지사항 객체 가져오기
        notice = Notice.objects.get(id=request.GET['id'])

        # 수정 페이지에 전달할 공지사항의 dict 타입의 데이터 생성
        context = {
            'notice': notice
        }

        # 공지사항 데이터를 가지고 수정 페이지로 이동
        return render(request, 'manager/manager-notice/manager-notice/manager-notice-modify.html', context)

    # 공지사항 수정 완료 이후의 뷰
    @transaction.atomic
    def post(self, request):
        # GET 방식으로 url에서 id를 가져옴
        notice_id = request.GET['id']

        # POST 방식으로 받은 id와 제목과 내용도 가져옴
        data = request.POST

        data = {
            'notice_title': data['notice-title'],
            'notice_content': data['notice-content']
        }

        # 가져온 id로 수정할 공지사항 조회
        notice = Notice.objects.get(id=notice_id)

        # 제목과 내용, 갱신 시간 변경하고 저장
        notice.notice_title = data['notice_title']
        notice.notice_content = data['notice_content']
        notice.updated_date = timezone.now()

        notice.save(update_fields=["notice_title", "notice_content", "updated_date"])

        # 기존 공지사항 정보 update 후, 공지사항 리스트 페이지로 redirect
        return redirect('manager-notice')


class DeleteNoticeView(View):
    # 공지사항 삭제를 위한 뷰
    def get(self, request):
        # 03/07 - 공지사항 리스트에서 체크한 게시물들을 한 번에 가져올 방법을 생각해보자
        # 삭제할 공지사항들의 id를 통해 해당 객체들을 가져옴(dict 타입)
        notices = Notice.objects.filter(id=request.GET['id'])

        # 위 공지사항들의 status를 0으로 만들어 화면에 뿌리지 않게 만들고, 변동 사항을 저장함
        for notice in notices:
            notice.notice_status = 0
            notice.updated_date = timezone.now()
            notice.save(update_fields=["notice_status", "updated_date"])

        # 상태 업데이트 후 공지사항 리스트 페이지로 redirect
        return redirect('manager-notice')


class DeleteManyNoticeView(APIView):
    # 한 번에 여러 개의 공지사항을 삭제(소프트 딜리트)하는 뷰
    def patch(self, request, notice_ids):
        # 요청 경로에 담긴 notice_ids를 콤마(,)를 기준으로 분리해서 list로 만듬
        notice_ids = notice_ids.split(',')

        # 위 list의 각 요소를 순회
        for notice_id in notice_ids:
            # 요소가 빈 문자열이 아닐 때만 tbl_notice에서 해당 id를 가진 객체를 가져옴
            if notice_id != '':
                notice = Notice.objects.get(id=notice_id)

                # 해당 객체의 status를 0으로 만들고, 변경 시간과 같이 저장
                notice.notice_status = 0
                notice.updated_date = timezone.now()
                notice.save(update_fields=["notice_status", "updated_date"])

        return Response('success')


# QnA 관리
class QnAManagementView(View):
    # QnA 내역 페이지 이동 뷰
    def get(self, request):
        # 현재 작성된 공지사항 및 QnA의 개수를 세서 dict 데이터로 통합
        notice_count = Notice.enabled_objects.count()
        qna_count = QnA.enabled_objects.count()

        context = {
            'notice_count': notice_count,
            'qna_count': qna_count
        }

        # 공지사항과 QnA 개수를 화면에 전달
        # QnA 내역을 가져오는 것은 별도의 API가 해줌
        return render(request, 'manager/manager-notice/manager-qna/manager-qna.html', context)


class WriteQnAView(View):
    # QnA 작성 페이지 이동 뷰
    def get(self, request):
        # 로그인 검사는 미들웨어가 해주기 때문에, 별도의 정보 필요 없이 바로 render
        return render(request, 'manager/manager-notice/manager-qna/manager-qna-compose.html')

    # QnA 작성 완료 이후의 뷰
    @transaction.atomic
    def post(self, request):
        # POST 방식으로 요청한 데이터를 가져옴
        qna_data = request.POST

        # 받아온 데이터에서 특정 정보(id, 제목, 내용)를 가져와서 dict 타입으로 저장
        data = {
            'qna_title': qna_data['qna-title'],
            'qna_content': qna_data['qna-content'],
        }

        # 받아온 데이터로 tbl_qna에 실행할 insert 쿼리 작성
        QnA.objects.create(**data)

        # 작성한 공지사항 저장 후, QnA 리스트 페이지로 redirect
        return redirect('manager-qna')


class UpdateQnAView(View):
    # QnA 수정 페이지 이동 뷰
    def get(self, request):
        # 수정 버튼에서 전달한 id를 통해 수정할 공지사항 객체 가져오기
        qna = QnA.objects.get(id=request.GET['id'])

        # 수정 페이지에 전달할 QnA의 dict 타입의 데이터 생성
        context = {
            'qna': qna
        }

        # QnA 데이터를 가지고 수정 페이지로 이동
        return render(request, 'manager/manager-notice/manager-qna/manager-qna-modify.html', context)

    # QnA 수정 완료 이후의 뷰
    @transaction.atomic
    def post(self, request):
        # GET 방식으로 url에서 id를 가져옴
        qna_id = request.GET['id']

        # POST 방식으로 받은 id와 제목과 내용도 가져옴
        data = request.POST

        data = {
            'qna_title': data['qna-title'],
            'qna_content': data['qna-content']
        }

        # 가져온 id로 수정할 QnA 조회
        qna = QnA.objects.get(id=qna_id)

        # 제목과 내용, 갱신 시간 변경하고 저장
        qna.qna_title = data['qna_title']
        qna.qna_content = data['qna_content']
        qna.updated_date = timezone.now()

        qna.save(update_fields=["qna_title", "qna_content", "updated_date"])

        # 기존 QnA 정보 update 후, QnA 리스트 페이지로 redirect
        return redirect('manager-qna')


class DeleteQnAView(View):
    # QnA 삭제를 위한 뷰
    def get(self, request):
        # 삭제할 QnA들의 id를 통해 해당 객체들을 가져옴
        qnas = QnA.objects.filter(id=request.GET['id'])

        # 위 QnA 전체를 소프트 딜리트(status = 0)
        for qna in qnas:
            qna.qna_status = 0
            qna.updated_date = timezone.now()
            qna.save(update_fields=["qna_status", "updated_date"])

        # 상태 업데이트 후 QnA 리스트 페이지로 redirect
        return redirect('manager-qna')


class DeleteManyQnAView(APIView):
    # 한 번에 여러 개의 QnA을 삭제(소프트 딜리트)하는 뷰
    def patch(self, request, qna_ids):
        # 요청 경로에 담긴 qna_ids를 콤마(,)를 기준으로 분리해서 list로 만듬
        qna_ids = qna_ids.split(',')

        # 위 list의 각 요소를 순회
        for qna_id in qna_ids:
            # 요소가 빈 문자열이 아닐 때만 tbl_qna에서 해당 id를 가진 객체를 가져옴
            if qna_id != '':
                qna = QnA.objects.get(id=qna_id)

                # 해당 객체의 status를 0으로 만들고, 변경 시간과 같이 저장
                qna.qna_status = 0
                qna.updated_date = timezone.now()
                qna.save(update_fields=["qna_status", "updated_date"])

        return Response('success')


# 강의 신고 내역 관리
class LectureReportManagementView(View):
    # 신고 내역 페이지 이동 뷰
    def get(self, request):
        # 강의, 거래, 일반 게시물(+댓글), 노하우 게시물(+댓글) 각각의 개수
        lecture_report_count = LectureReport.object.count()  # 강의
        trade_report_count = TradeReport.object.count()  # 거래
        post_report_count = PostReport.object.count()  # 일반 게시물
        post_reply_report_count = PostReplyReport.object.count()  # 일반 게시물 댓글
        knowhow_report_count = KnowhowReport.object.count()  # 노하우 게시물
        knowhow_reply_report_count = KnowhowReplyReport.object.count()  # 노하우 게시물 댓글

        # 위의 모든 정보를 화면으로 보내기 전, dict 형식으로 묶어줌
        context = {
            'lecture_report_count': lecture_report_count,
            'trade_report_count': trade_report_count,
            'post_report_count': post_report_count,
            'post_reply_report_count': post_reply_report_count,
            'knowhow_report_count': knowhow_report_count,
            'knowhow_reply_report_count': knowhow_reply_report_count
        }

        return render(request, 'manager/report/lecture-report.html', context)


# 거래 신고 내역 관리
class TradeReportManagementView(View):
    # 신고 내역 페이지 이동 뷰
    def get(self, request):
        # 강의, 거래, 일반 게시물(+댓글), 노하우 게시물(+댓글) 각각의 개수
        lecture_report_count = LectureReport.object.count()  # 강의
        trade_report_count = TradeReport.object.count()  # 거래
        post_report_count = PostReport.object.count()  # 일반 게시물
        post_reply_report_count = PostReplyReport.object.count()  # 일반 게시물 댓글
        knowhow_report_count = KnowhowReport.object.count()  # 노하우 게시물
        knowhow_reply_report_count = KnowhowReplyReport.object.count()  # 노하우 게시물 댓글

        # 위의 모든 정보를 화면으로 보내기 전, dict 형식으로 묶어줌
        context = {
            'lecture_report_count': lecture_report_count,
            'trade_report_count': trade_report_count,
            'post_report_count': post_report_count,
            'post_reply_report_count': post_reply_report_count,
            'knowhow_report_count': knowhow_report_count,
            'knowhow_reply_report_count': knowhow_reply_report_count
        }

        return render(request, 'manager/report/trade-report.html', context)


# 일반 게시물 신고 내역 관리
class PostReportManagementView(View):
    # 신고 내역 페이지 이동 뷰
    def get(self, request):
        # 강의, 거래, 일반 게시물(+댓글), 노하우 게시물(+댓글) 각각의 개수
        lecture_report_count = LectureReport.object.count()  # 강의
        trade_report_count = TradeReport.object.count()  # 거래
        post_report_count = PostReport.object.count()  # 일반 게시물
        post_reply_report_count = PostReplyReport.object.count()  # 일반 게시물 댓글
        knowhow_report_count = KnowhowReport.object.count()  # 노하우 게시물
        knowhow_reply_report_count = KnowhowReplyReport.object.count()  # 노하우 게시물 댓글

        # 위의 모든 정보를 화면으로 보내기 전, dict 형식으로 묶어줌
        context = {
            'lecture_report_count': lecture_report_count,
            'trade_report_count': trade_report_count,
            'post_report_count': post_report_count,
            'post_reply_report_count': post_reply_report_count,
            'knowhow_report_count': knowhow_report_count,
            'knowhow_reply_report_count': knowhow_reply_report_count
        }

        return render(request, 'manager/report/post-report.html', context)


# 일반 댓글 신고 내역 관리
class PostReplyReportManagementView(View):
    # 신고 내역 페이지 이동 뷰
    def get(self, request):
        # 강의, 거래, 일반 게시물(+댓글), 노하우 게시물(+댓글) 각각의 개수
        lecture_report_count = LectureReport.object.count()  # 강의
        trade_report_count = TradeReport.object.count()  # 거래
        post_report_count = PostReport.object.count()  # 일반 게시물
        post_reply_report_count = PostReplyReport.object.count()  # 일반 게시물 댓글
        knowhow_report_count = KnowhowReport.object.count()  # 노하우 게시물
        knowhow_reply_report_count = KnowhowReplyReport.object.count()  # 노하우 게시물 댓글

        # 위의 모든 정보를 화면으로 보내기 전, dict 형식으로 묶어줌
        context = {
            'lecture_report_count': lecture_report_count,
            'trade_report_count': trade_report_count,
            'post_report_count': post_report_count,
            'post_reply_report_count': post_reply_report_count,
            'knowhow_report_count': knowhow_report_count,
            'knowhow_reply_report_count': knowhow_reply_report_count
        }

        return render(request, 'manager/report/post-reply-report.html', context)


# 노하우 신고 내역 관리
class KnowhowReportManagementView(View):
    # 신고 내역 페이지 이동 뷰
    def get(self, request):
        # 강의, 거래, 일반 게시물(+댓글), 노하우 게시물(+댓글) 각각의 개수
        lecture_report_count = LectureReport.object.count()  # 강의
        trade_report_count = TradeReport.object.count()  # 거래
        post_report_count = PostReport.object.count()  # 일반 게시물
        post_reply_report_count = PostReplyReport.object.count()  # 일반 게시물 댓글
        knowhow_report_count = KnowhowReport.object.count()  # 노하우 게시물
        knowhow_reply_report_count = KnowhowReplyReport.object.count()  # 노하우 게시물 댓글

        # 위의 모든 정보를 화면으로 보내기 전, dict 형식으로 묶어줌
        context = {
            'lecture_report_count': lecture_report_count,
            'trade_report_count': trade_report_count,
            'post_report_count': post_report_count,
            'post_reply_report_count': post_reply_report_count,
            'knowhow_report_count': knowhow_report_count,
            'knowhow_reply_report_count': knowhow_reply_report_count
        }

        return render(request, 'manager/report/knowhow-report.html', context)


# 노하우 댓글 신고 내역 관리
class KnowhowReplyReportManagementView(View):
    # 신고 내역 페이지 이동 뷰
    def get(self, request):
        # 강의, 거래, 일반 게시물(+댓글), 노하우 게시물(+댓글) 각각의 개수
        lecture_report_count = LectureReport.object.filter(report_status=1).count()  # 강의
        trade_report_count = TradeReport.object.filter(report_status=1).count()  # 거래
        post_report_count = PostReport.object.filter(report_status=1).count()  # 일반 게시물
        post_reply_report_count = PostReplyReport.object.filter(report_status=1).count()  # 일반 게시물 댓글
        knowhow_report_count = KnowhowReport.object.filter(report_status=1).count()  # 노하우 게시물
        knowhow_reply_report_count = KnowhowReplyReport.object.filter(report_status=1).count()  # 노하우 게시물 댓글

        # 위의 모든 정보를 화면으로 보내기 전, dict 형식으로 묶어줌
        context = {
            'lecture_report_count': lecture_report_count,
            'trade_report_count': trade_report_count,
            'post_report_count': post_report_count,
            'post_reply_report_count': post_reply_report_count,
            'knowhow_report_count': knowhow_report_count,
            'knowhow_reply_report_count': knowhow_reply_report_count
        }

        return render(request, 'manager/report/knowhow-reply-report.html', context)


class LectureReportAPI(APIView):
    # 강의 신고 API 뷰
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 신고 내역 수
        row_count = 10

        # 한 페이지에 표시할 신고 내역들을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 신고 사유 or 신고자 닉네임 or 신고 대상의 제목을 검색
        if keyword:
            condition |= Q(report_content__icontains=keyword)
            condition |= Q(report_member__icontains=keyword)
            condition |= Q(report_target__icontains=keyword)

        # 신고 내역 표시에 필요한 컬럼들
        columns = [
            'id',
            'report_content',  # 신고 내용
            'report_member',  # 신고자
            'report_status',  # 1 - 접수됨, 0 - 삭제됨
            'report_target',  # 신고 대상 - 강의
            'created_date',  # 신고 일자
        ]

        # 신고 내역 조회
        lecture_reports = LectureReport.object \
            .annotate(report_member=F('member__member_name'),
                      report_target=F('lecture__lecture_title')) \
            .values(*columns).filter(condition, id__isnull=False, report_status=1)

        # 각 신고 내역의 신고 일자를 "YYYY.MM.DD" 형식의 문자열로 변환
        for lecture_report in lecture_reports:
            lecture_report['created_date'] = lecture_report['created_date'].strftime('%Y.%m.%d')

        # 신고 내역 수
        total = lecture_reports.count()

        # 각 유형 별 신고 내역 수
        lecture_report_count = LectureReport.object.filter(report_status=1).count()  # 강의
        trade_report_count = TradeReport.object.filter(report_status=1).count()  # 거래
        post_report_count = PostReport.object.filter(report_status=1).count()  # 일반 게시물
        post_reply_report_count = PostReplyReport.object.filter(report_status=1).count()  # 일반 게시물 댓글
        knowhow_report_count = KnowhowReport.object.filter(report_status=1).count()  # 노하우 게시물
        knowhow_reply_report_count = KnowhowReplyReport.object.filter(report_status=1).count()  # 노하우 게시물 댓글

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
            'lectureReports': lecture_report_count,
            'tradeReports': trade_report_count,
            'postReports': post_report_count,
            'postReplyReports': post_reply_report_count,
            'knowhowReports': knowhow_report_count,
            'knowhowReplyReports': knowhow_reply_report_count
        }

        # 신고 내역 목록을 QuerySet -> list 타입으로 변경하고, 신고 내역 10개씩 슬라이싱(페이지 하나)
        lecture_reports = list(lecture_reports[offset:limit])

        # 신고 내역 목록의 맨 뒤에 페이지네이션 정보 추가
        lecture_reports.append(page_info)

        # 요청한 강의 신고 내역 및 페이지네이션에 사용할 정보 반환
        return Response(lecture_reports)


class TradeReportAPI(APIView):
    # 거래 신고 API 뷰
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 신고 내역 수
        row_count = 10

        # 한 페이지에 표시할 신고 내역들을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 신고 사유 or 신고자 닉네임 or 신고 대상의 제목을 검색
        if keyword:
            condition |= Q(report_content__icontains=keyword)
            condition |= Q(report_member__icontains=keyword)
            condition |= Q(report_target__icontains=keyword)

        # 신고 내역 표시에 필요한 컬럼들
        columns = [
            'id',
            'report_content',  # 신고 내용
            'report_member',  # 신고자
            'report_status',  # 1 - 접수됨, 0 - 삭제됨
            'report_target',  # 신고 대상 - 거래
            'created_date',  # 신고 일자
        ]

        # 신고 내역 조회
        trade_reports = TradeReport.object \
            .annotate(report_member=F('member__member_name'),
                      report_target=F('trade__trade_title')) \
            .values(*columns).filter(condition, id__isnull=False, report_status=1)

        # 각 신고 내역의 신고 일자를 "YYYY.MM.DD" 형식의 문자열로 변환
        for trade_report in trade_reports:
            trade_report['created_date'] = trade_report['created_date'].strftime('%Y.%m.%d')

        # 신고 내역 수
        total = trade_reports.count()

        # 각 유형 별 신고 내역 수
        lecture_report_count = LectureReport.object.filter(report_status=1).count()  # 강의
        trade_report_count = TradeReport.object.filter(report_status=1).count()  # 거래
        post_report_count = PostReport.object.filter(report_status=1).count()  # 일반 게시물
        post_reply_report_count = PostReplyReport.object.filter(report_status=1).count()  # 일반 게시물 댓글
        knowhow_report_count = KnowhowReport.object.filter(report_status=1).count()  # 노하우 게시물
        knowhow_reply_report_count = KnowhowReplyReport.object.filter(report_status=1).count()  # 노하우 게시물 댓글

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
            'lectureReports': lecture_report_count,
            'tradeReports': trade_report_count,
            'postReports': post_report_count,
            'postReplyReports': post_reply_report_count,
            'knowhowReports': knowhow_report_count,
            'knowhowReplyReports': knowhow_reply_report_count
        }

        # 신고 내역 목록을 QuerySet -> list 타입으로 변경하고, 신고 내역 10개씩 슬라이싱(페이지 하나)
        trade_reports = list(trade_reports[offset:limit])

        # 신고 목록의 맨 뒤에 페이지네이션 정보 추가
        trade_reports.append(page_info)

        # 요청한 거래 신고 내역 및 페이지네이션에 사용할 정보 반환
        return Response(trade_reports)


class PostReportAPI(APIView):
    # 일반 게시물 신고 API 뷰
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 신고 내역 수
        row_count = 10

        # 한 페이지에 표시할 신고 내역들을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 신고 사유 or 신고자 닉네임 or 신고 대상의 제목을 검색
        if keyword:
            condition |= Q(report_content__icontains=keyword)
            condition |= Q(report_member__icontains=keyword)
            condition |= Q(report_target__icontains=keyword)

        # 신고 내역 표시에 필요한 컬럼들
        columns = [
            'id',
            'report_content',  # 신고 내용
            'report_member',  # 신고자
            'report_status',  # 1 - 접수됨, 0 - 삭제됨
            'report_target',  # 신고 대상 - 일반 게시물
            'created_date',  # 신고 일자
        ]

        # 신고 내역 조회
        post_reports = PostReport.object \
            .annotate(report_member=F('member__member_name'),
                      report_target=F('post__post_title')) \
            .values(*columns).filter(condition, id__isnull=False, report_status=1)

        # 각 신고 내역의 신고 일자를 "YYYY.MM.DD" 형식의 문자열로 변환
        for post_report in post_reports:
            post_report['created_date'] = post_report['created_date'].strftime('%Y.%m.%d')

        # 신고 내역 수
        total = post_reports.count()

        # 각 유형 별 신고 내역 수
        lecture_report_count = LectureReport.object.filter(report_status=1).count()  # 강의
        trade_report_count = TradeReport.object.filter(report_status=1).count()  # 거래
        post_report_count = PostReport.object.filter(report_status=1).count()  # 일반 게시물
        post_reply_report_count = PostReplyReport.object.filter(report_status=1).count()  # 일반 게시물 댓글
        knowhow_report_count = KnowhowReport.object.filter(report_status=1).count()  # 노하우 게시물
        knowhow_reply_report_count = KnowhowReplyReport.object.filter(report_status=1).count()  # 노하우 게시물 댓글

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
            'lectureReports': lecture_report_count,
            'tradeReports': trade_report_count,
            'postReports': post_report_count,
            'postReplyReports': post_reply_report_count,
            'knowhowReports': knowhow_report_count,
            'knowhowReplyReports': knowhow_reply_report_count
        }

        # 신고 내역 목록을 QuerySet -> list 타입으로 변경하고, 신고 내역 10개씩 슬라이싱(페이지 하나)
        post_reports = list(post_reports[offset:limit])

        # 신고 목록의 맨 뒤에 페이지네이션 정보 추가
        post_reports.append(page_info)

        # 요청한 일반 게시물 신고 내역 및 페이지네이션에 사용할 정보 반환
        return Response(post_reports)


class PostReplyReportAPI(APIView):
    # 일반 게시물 댓글 신고 API 뷰
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 신고 내역 수
        row_count = 10

        # 한 페이지에 표시할 신고 내역들을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 신고 사유 or 신고자 닉네임 or 신고 대상의 제목을 검색
        if keyword:
            condition |= Q(report_content__icontains=keyword)
            condition |= Q(report_member__icontains=keyword)
            condition |= Q(report_target__icontains=keyword)

        # 신고 내역 표시에 필요한 컬럼들
        columns = [
            'id',
            'report_content',  # 신고 내용
            'report_member',  # 신고자
            'report_status',  # 1 - 접수됨, 0 - 삭제됨
            'report_target',  # 신고 대상 - 일반 게시물 댓글
            'created_date',  # 신고 일자
        ]

        # 신고 내역 조회
        post_reply_reports = PostReplyReport.object \
            .annotate(report_member=F('member__member_name'),
                      report_target=F('post_reply__post_reply_content')) \
            .values(*columns).filter(condition, id__isnull=False, report_status=1)

        # 각 신고 내역의 신고 일자를 "YYYY.MM.DD" 형식의 문자열로 변환
        for post_reply_report in post_reply_reports:
            post_reply_report['created_date'] = post_reply_report['created_date'].strftime('%Y.%m.%d')

        # 신고 내역 수
        total = post_reply_reports.count()

        # 각 유형 별 신고 내역 수
        lecture_report_count = LectureReport.object.filter(report_status=1).count()  # 강의
        trade_report_count = TradeReport.object.filter(report_status=1).count()  # 거래
        post_report_count = PostReport.object.filter(report_status=1).count()  # 일반 게시물
        post_reply_report_count = PostReplyReport.object.filter(report_status=1).count()  # 일반 게시물 댓글
        knowhow_report_count = KnowhowReport.object.filter(report_status=1).count()  # 노하우 게시물
        knowhow_reply_report_count = KnowhowReplyReport.object.filter(report_status=1).count()  # 노하우 게시물 댓글

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
            'lectureReports': lecture_report_count,
            'tradeReports': trade_report_count,
            'postReports': post_report_count,
            'postReplyReports': post_reply_report_count,
            'knowhowReports': knowhow_report_count,
            'knowhowReplyReports': knowhow_reply_report_count
        }

        # 신고 내역 목록을 QuerySet -> list 타입으로 변경하고, 신고 내역 10개씩 슬라이싱(페이지 하나)
        post_reply_reports = list(post_reply_reports[offset:limit])

        # 신고 목록의 맨 뒤에 페이지네이션 정보 추가
        post_reply_reports.append(page_info)

        # 요청한 일반 게시물 댓글 신고 내역 및 페이지네이션에 사용할 정보 반환
        return Response(post_reply_reports)


class KnowhowReportAPI(APIView):
    # 노하우 게시물 신고 API 뷰
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 신고 내역 수
        row_count = 10

        # 한 페이지에 표시할 신고 내역들을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 신고 사유 or 신고자 닉네임 or 신고 대상의 제목을 검색
        if keyword:
            condition |= Q(report_content__icontains=keyword)
            condition |= Q(report_member__icontains=keyword)
            condition |= Q(report_target__icontains=keyword)

        # 신고 내역 표시에 필요한 컬럼들
        columns = [
            'id',
            'report_content',  # 신고 내용
            'report_member',  # 신고자
            'report_status',  # 1 - 접수됨, 0 - 삭제됨
            'report_target',  # 신고 대상 - 노하우 게시물
            'created_date',  # 신고 일자
        ]

        # 신고 내역 조회
        knowhow_reports = KnowhowReport.object \
            .annotate(report_member=F('member__member_name'),
                      report_target=F('knowhow__knowhow_title')) \
            .values(*columns).filter(condition, id__isnull=False, report_status=1)

        # 각 신고 내역의 신고 일자를 "YYYY.MM.DD" 형식의 문자열로 변환
        for knowhow_report in knowhow_reports:
            knowhow_report['created_date'] = knowhow_report['created_date'].strftime('%Y.%m.%d')

        # 신고 내역 수
        total = knowhow_reports.count()

        # 각 유형 별 신고 내역 수
        lecture_report_count = LectureReport.object.filter(report_status=1).count()  # 강의
        trade_report_count = TradeReport.object.filter(report_status=1).count()  # 거래
        post_report_count = PostReport.object.filter(report_status=1).count()  # 일반 게시물
        post_reply_report_count = PostReplyReport.object.filter(report_status=1).count()  # 일반 게시물 댓글
        knowhow_report_count = KnowhowReport.object.filter(report_status=1).count()  # 노하우 게시물
        knowhow_reply_report_count = KnowhowReplyReport.object.filter(report_status=1).count()  # 노하우 게시물 댓글

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
            'lectureReports': lecture_report_count,
            'tradeReports': trade_report_count,
            'postReports': post_report_count,
            'postReplyReports': post_reply_report_count,
            'knowhowReports': knowhow_report_count,
            'knowhowReplyReports': knowhow_reply_report_count
        }

        # 신고 내역 목록을 QuerySet -> list 타입으로 변경하고, 신고 내역 10개씩 슬라이싱(페이지 하나)
        knowhow_reports = list(knowhow_reports[offset:limit])

        # 신고 목록의 맨 뒤에 페이지네이션 정보 추가
        knowhow_reports.append(page_info)

        # 요청한 노하우 게시물 신고 내역 및 페이지네이션에 사용할 정보 반환
        return Response(knowhow_reports)


class KnowhowReplyReportAPI(APIView):
    # 노하우 게시물 댓글 신고 API 뷰
    def get(self, request):
        # 쿼리 스트링에서 검색 키워드와 페이지 값 받아오기
        keyword = request.GET.get('keyword', '')
        page = int(request.GET.get('page', 1))

        # 한 페이지에 띄울 신고 내역 수
        row_count = 10

        # 한 페이지에 표시할 신고 내역들을 슬라이싱 하기 위한 변수들
        offset = (page - 1) * row_count
        limit = page * row_count

        # 검색 조건식 선언
        condition = Q()

        # keyword로 뭐라도 받았다면
        # keyword가 포함된 신고 사유 or 신고자 닉네임 or 신고 대상의 제목을 검색
        if keyword:
            condition |= Q(report_content__icontains=keyword)
            condition |= Q(report_member__icontains=keyword)
            condition |= Q(report_target__icontains=keyword)

        # 신고 내역 표시에 필요한 컬럼들
        columns = [
            'id',
            'report_content',  # 신고 내용
            'report_member',  # 신고자
            'report_status',  # 1 - 접수됨, 0 - 삭제됨
            'report_target',  # 신고 대상 - 노하우 게시물 댓글
            'created_date',  # 신고 일자
        ]

        # 신고 내역 조회
        knowhow_reply_reports = KnowhowReplyReport.object \
            .annotate(report_member=F('member__member_name'),
                      report_target=F('knowhow_reply__knowhow_reply_content')) \
            .values(*columns).filter(condition, id__isnull=False, report_status=1)

        # 각 신고 내역의 신고 일자를 "YYYY.MM.DD" 형식의 문자열로 변환
        for knowhow_reply_report in knowhow_reply_reports:
            knowhow_reply_report['created_date'] = knowhow_reply_report['created_date'].strftime('%Y.%m.%d')

        # 신고 내역 수
        total = knowhow_reply_reports.count()

        # 각 유형 별 신고 내역 수
        lecture_report_count = LectureReport.object.filter(report_status=1).count()  # 강의
        trade_report_count = TradeReport.object.filter(report_status=1).count()  # 거래
        post_report_count = PostReport.object.filter(report_status=1).count()  # 일반 게시물
        post_reply_report_count = PostReplyReport.object.filter(report_status=1).count()  # 일반 게시물 댓글
        knowhow_report_count = KnowhowReport.object.filter(report_status=1).count()  # 노하우 게시물
        knowhow_reply_report_count = KnowhowReplyReport.object.filter(report_status=1).count()  # 노하우 게시물 댓글

        # 페이지네이션에 필요한 정보들
        page_count = 5  # 화면에 표시할 페이지 숫자 버튼의 최대 개수

        end_page = math.ceil(page / page_count) * page_count  # 화면에 표시할 페이지 숫자 버튼 중 마지막 페이지
        start_page = end_page - page_count + 1  # 화면에 표시할 페이지 숫자 버튼 중 첫 페이지
        real_end = math.ceil(total / row_count)  # 전체 리스트의 마지막 페이지

        # end_page의 값이 real_end 보다 커지지 않게 조정
        end_page = real_end if end_page > real_end else end_page

        # end_page의 값이 0보다 작아지지 않게 조정
        if end_page == 0:
            end_page = 1

        # 페이지네이션에 사용할 정보 완성
        page_info = {
            'totalCount': total,
            'startPage': start_page,
            'endPage': end_page,
            'page': page,
            'realEnd': real_end,
            'pageCount': page_count,
            'lectureReports': lecture_report_count,
            'tradeReports': trade_report_count,
            'postReports': post_report_count,
            'postReplyReports': post_reply_report_count,
            'knowhowReports': knowhow_report_count,
            'knowhowReplyReports': knowhow_reply_report_count
        }

        # 신고 내역 목록을 QuerySet -> list 타입으로 변경하고, 신고 내역 10개씩 슬라이싱(페이지 하나)
        knowhow_reply_reports = list(knowhow_reply_reports[offset:limit])

        # 신고 목록의 맨 뒤에 페이지네이션 정보 추가
        knowhow_reply_reports.append(page_info)

        # 요청한 노하우 게시물 댓글 신고 내역 및 페이지네이션에 사용할 정보 반환
        return Response(knowhow_reply_reports)


# 강의 신고 승인 및 삭제를 담당하는 API 뷰
class LectureReportAdjustAPI(APIView):
    def patch(self, request, report_ids):
        # 요청 경로에 담긴 아이디를 콤마(,)를 기준으로 분리해서 list로 만듬
        report_ids = report_ids.split(',')

        # 위 list의 각 요소를 순회
        for report_id in report_ids:
            # 요소가 빈 문자열이 아닐 때만 테이블에서 해당 id를 가진 객체를 가져옴
            if report_id != '':
                report = LectureReport.object.get(id=report_id)

                # 해당 객체의 status를 0으로 만들고, 변경 시간과 같이 저장
                report.report_status = 0
                report.updated_date = timezone.now()
                report.save(update_fields=["report_status", "updated_date"])

        return Response('success')

    def delete(self, request, report_ids):
        # 요청 경로에 담긴 아이디를 콤마(,)를 기준으로 분리해서 list로 만듬
        report_ids = report_ids.split(',')

        # 위 list의 각 요소를 순회
        for report_id in report_ids:
            # 요소가 빈 문자열이 아닐 때만 테이블에서 해당 id를 가진 객체를 가져와서 삭제
            if report_id != '':
                LectureReport.object.get(id=report_id).delete()

        return Response('success')


# 거래 신고 승인 및 삭제를 담당하는 API 뷰
class TradeReportAdjustAPI(APIView):
    def patch(self, request, report_ids):
        # 요청 경로에 담긴 아이디를 콤마(,)를 기준으로 분리해서 list로 만듬
        report_ids = report_ids.split(',')

        # 위 list의 각 요소를 순회
        for report_id in report_ids:
            # 요소가 빈 문자열이 아닐 때만 테이블에서 해당 id를 가진 객체를 가져옴
            if report_id != '':
                report = TradeReport.object.get(id=report_id)

                # 해당 객체의 status를 0으로 만들고, 변경 시간과 같이 저장
                report.report_status = 0
                report.updated_date = timezone.now()
                report.save(update_fields=["report_status", "updated_date"])

        return Response('success')

    def delete(self, request, report_ids):
        # 요청 경로에 담긴 아이디를 콤마(,)를 기준으로 분리해서 list로 만듬
        report_ids = report_ids.split(',')

        # 위 list의 각 요소를 순회
        for report_id in report_ids:
            # 요소가 빈 문자열이 아닐 때만 테이블에서 해당 id를 가진 객체를 가져와서 삭제
            if report_id != '':
                TradeReport.object.get(id=report_id).delete()

        return Response('success')


# 일반 게시물 신고 승인 및 삭제를 담당하는 API 뷰
class PostReportAdjustAPI(APIView):
    def patch(self, request, report_ids):
        # 요청 경로에 담긴 아이디를 콤마(,)를 기준으로 분리해서 list로 만듬
        report_ids = report_ids.split(',')

        # 위 list의 각 요소를 순회
        for report_id in report_ids:
            # 요소가 빈 문자열이 아닐 때만 테이블에서 해당 id를 가진 객체를 가져옴
            if report_id != '':
                report = PostReport.object.get(id=report_id)

                # 해당 객체의 status를 0으로 만들고, 변경 시간과 같이 저장
                report.report_status = 0
                report.updated_date = timezone.now()
                report.save(update_fields=["report_status", "updated_date"])

        return Response('success')

    def delete(self, request, report_ids):
        # 요청 경로에 담긴 아이디를 콤마(,)를 기준으로 분리해서 list로 만듬
        report_ids = report_ids.split(',')

        # 위 list의 각 요소를 순회
        for report_id in report_ids:
            # 요소가 빈 문자열이 아닐 때만 테이블에서 해당 id를 가진 객체를 가져와서 삭제
            if report_id != '':
                PostReport.object.get(id=report_id).delete()

        return Response('success')


# 일반 게시물 댓글 신고 승인 및 삭제를 담당하는 API 뷰
class PostReplyReportAdjustAPI(APIView):
    def patch(self, request, report_ids):
        # 요청 경로에 담긴 아이디를 콤마(,)를 기준으로 분리해서 list로 만듬
        report_ids = report_ids.split(',')

        # 위 list의 각 요소를 순회
        for report_id in report_ids:
            # 요소가 빈 문자열이 아닐 때만 테이블에서 해당 id를 가진 객체를 가져옴
            if report_id != '':
                report = PostReplyReport.object.get(id=report_id)

                # 해당 객체의 status를 0으로 만들고, 변경 시간과 같이 저장
                report.report_status = 0
                report.updated_date = timezone.now()
                report.save(update_fields=["report_status", "updated_date"])

        return Response('success')

    def delete(self, request, report_ids):
        # 요청 경로에 담긴 아이디를 콤마(,)를 기준으로 분리해서 list로 만듬
        report_ids = report_ids.split(',')

        # 위 list의 각 요소를 순회
        for report_id in report_ids:
            # 요소가 빈 문자열이 아닐 때만 테이블에서 해당 id를 가진 객체를 가져와서 삭제
            if report_id != '':
                post_report_reply = PostReplyReport.object.filter(id=report_id).values().first()
                report_reply_id = post_report_reply.get('post_reply_id')
                post_reply = PostReply.objects.filter(id=report_reply_id).values().first()

                new_sentence = [post_reply['post_reply_content']]
                data = {
                    'comment': new_sentence[0],
                    'target': 1,
                }
                print(data)
                # profanityDetectionModel(new_sentence)
                # AiPostReply.objects.create(**data)
                PostReplyReport.object.get(id=report_id).delete()
                PostReply.objects.get(id=report_reply_id).delete()


                # try:
                #     with transaction.atomic():
                #         with connection.cursor() as cursor:
                #             cursor.execute('SET foreign_key_checks = 0;')
                #             PostReply.objects.get(id=report_reply_id).delete()
                #             PostReplyReport.objects.get(id=report_id).delete()
                #             cursor.execute('SET foreign_key_checks = 1;')
                # except Exception as e:
                #     print(f"Error occurred: {e}")

        return Response('success')


# 노하우 게시물 신고 승인 및 삭제를 담당하는 API 뷰
class KnowhowReportAdjustAPI(APIView):
    def patch(self, request, report_ids):
        # 요청 경로에 담긴 아이디를 콤마(,)를 기준으로 분리해서 list로 만듬
        report_ids = report_ids.split(',')

        # 위 list의 각 요소를 순회
        for report_id in report_ids:
            # 요소가 빈 문자열이 아닐 때만 테이블에서 해당 id를 가진 객체를 가져옴
            if report_id != '':
                report = KnowhowReport.object.get(id=report_id)

                # 해당 객체의 status를 0으로 만들고, 변경 시간과 같이 저장
                report.report_status = 0
                report.updated_date = timezone.now()
                report.save(update_fields=["report_status", "updated_date"])

        return Response('success')

    def delete(self, request, report_ids):
        # 요청 경로에 담긴 아이디를 콤마(,)를 기준으로 분리해서 list로 만듬
        report_ids = report_ids.split(',')

        # 위 list의 각 요소를 순회
        for report_id in report_ids:
            # 요소가 빈 문자열이 아닐 때만 테이블에서 해당 id를 가진 객체를 가져와서 삭제
            if report_id != '':
                KnowhowReport.object.get(id=report_id).delete()

        return Response('success')


# 노하우 게시물 신고 승인 및 삭제를 담당하는 API 뷰
class KnowhowReplyReportAdjustAPI(APIView):
    def patch(self, request, report_ids):
        # 요청 경로에 담긴 아이디를 콤마(,)를 기준으로 분리해서 list로 만듬
        report_ids = report_ids.split(',')

        # 위 list의 각 요소를 순회
        for report_id in report_ids:
            # 요소가 빈 문자열이 아닐 때만 테이블에서 해당 id를 가진 객체를 가져옴
            if report_id != '':
                report = KnowhowReplyReport.object.get(id=report_id)

                # 해당 객체의 status를 0으로 만들고, 변경 시간과 같이 저장
                report.report_status = 0
                report.updated_date = timezone.now()
                report.save(update_fields=["report_status", "updated_date"])

        return Response('success')

    def delete(self, request, report_ids):
        # 요청 경로에 담긴 아이디를 콤마(,)를 기준으로 분리해서 list로 만듬
        report_ids = report_ids.split(',')

        # 위 list의 각 요소를 순회
        for report_id in report_ids:
            # 요소가 빈 문자열이 아닐 때만 테이블에서 해당 id를 가진 객체를 가져옴
            if report_id != '':
                KnowhowReplyReport.object.get(id=report_id).delete()

        return Response('success')
